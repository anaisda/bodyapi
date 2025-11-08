from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import requests
import json
import os
import datetime
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class MeasurementCorrector:
    """Handles measurement corrections with conservative, body-type-aware methods."""
    
    def __init__(self):
        # CONSERVATIVE correction factors - small adjustments only
        # All factors should be close to 1.0 to avoid dramatic changes
        self.correction_factors = {
            'chest_bust': {
                'base_factor': 1.02,  # Slight increase (AI tends to underestimate slightly)
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 1.01,
                    'regular': 1.02,
                    'loose': 1.03
                },
                'bmi_adjustments': {
                    'underweight': 0.98,   # Smaller chest for underweight
                    'normal': 1.00,
                    'overweight': 1.02,
                    'obese': 1.03
                }
            },
            'waist': {
                'base_factor': 1.03,  # Slight increase
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 1.02,
                    'regular': 1.04,
                    'loose': 1.05
                },
                'bmi_adjustments': {
                    'underweight': 0.98,
                    'normal': 1.00,
                    'overweight': 1.03,
                    'obese': 1.05
                }
            },
            'hips': {
                'base_factor': 0.96,  # Slight decrease (AI sometimes overestimates)
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 0.99,
                    'regular': 0.98,
                    'loose': 0.97
                },
                'bmi_adjustments': {
                    'underweight': 1.02,   # Wider hip-to-chest ratio for underweight
                    'normal': 1.00,
                    'overweight': 0.99,
                    'obese': 0.98
                },
                'gender_adjustments': {
                    'male': 0.98,      # Slightly narrower hips
                    'female': 1.02     # Slightly wider hips
                }
            }
        }
        
        # Soft validation ranges - for warnings only, not forced corrections
        # These are WIDE ranges that accommodate different body types
        self.proportion_suggestions = {
            'waist_to_chest': {'min': 0.70, 'max': 1.30},
            'hip_to_chest': {
                'male': {'min': 0.65, 'max': 1.20},      # Wide range for all male body types
                'female': {'min': 0.75, 'max': 1.15}     # Wide range for all female body types
            },
            'hip_to_waist': {
                'male': {'min': 0.80, 'max': 1.25},
                'female': {'min': 0.85, 'max': 1.30}
            }
        }
        
        # Absolute safety limits (prevent physically impossible measurements)
        self.safety_limits = {
            'chest_bust': {'min': 50, 'max': 200},
            'waist': {'min': 40, 'max': 200},
            'hips': {'min': 50, 'max': 200}
        }
    
    def get_bmi_category(self, height_cm, weight_kg):
        """Calculate BMI category."""
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            return 'underweight', bmi
        elif bmi < 25:
            return 'normal', bmi
        elif bmi < 30:
            return 'overweight', bmi
        else:
            return 'obese', bmi
    
    def get_clothing_category(self, clothing_description):
        """Normalize clothing description to standard categories."""
        clothing_lower = clothing_description.lower()
        
        if any(term in clothing_lower for term in ['skin-tight', 'athletic', 'compression']):
            return 'skin-tight'
        elif any(term in clothing_lower for term in ['fitted', 'slim']):
            return 'fitted'
        elif any(term in clothing_lower for term in ['regular', 't-shirt', 'shirt']):
            return 'regular'
        else:
            return 'loose'
    
    def apply_corrections(self, measurements_data, subject_profile):
        """Apply conservative, body-type-aware corrections to measurements."""
        try:
            corrections_applied = []
            warnings = []
            circumferences = measurements_data.get("measurements", {}).get("circumferences_cm", {})
            
            # Get subject characteristics
            height = float(subject_profile.get("height_cm", 170))
            weight = float(subject_profile.get("weight_kg", 70))
            gender = subject_profile.get("gender", "Male").lower()
            clothing = subject_profile.get("clothing_type", "fitted")
            
            bmi_category, bmi_value = self.get_bmi_category(height, weight)
            clothing_category = self.get_clothing_category(clothing)
            
            # Apply conservative corrections to each measurement
            for measurement_name in ['chest_bust', 'waist', 'hips']:
                if measurement_name in circumferences and "value" in circumferences[measurement_name]:
                    original_value = float(circumferences[measurement_name]["value"])
                    
                    # Apply correction
                    corrected_value = self._calculate_corrected_value(
                        original_value, measurement_name, bmi_category, 
                        clothing_category, gender
                    )
                    
                    # Apply safety limits
                    limits = self.safety_limits[measurement_name]
                    corrected_value = max(limits['min'], min(limits['max'], corrected_value))
                    
                    # Only update if change is significant (>2cm) and reasonable
                    if abs(corrected_value - original_value) > 2.0:
                        circumferences[measurement_name]["value"] = round(corrected_value, 1)
                        circumferences[measurement_name]["correction_applied"] = True
                        circumferences[measurement_name]["original_value"] = round(original_value, 1)
                        circumferences[measurement_name]["correction_notes"] = f"Conservative adjustment: {original_value:.1f} → {corrected_value:.1f} cm"
                        corrections_applied.append(
                            f"{measurement_name}: {original_value:.1f} → {corrected_value:.1f} cm"
                        )
                    else:
                        circumferences[measurement_name]["correction_applied"] = False
                        circumferences[measurement_name]["correction_notes"] = "No significant correction needed"
            
            # Check proportions (generate warnings only, no forced corrections)
            proportion_warnings = self._check_proportions(circumferences, gender)
            warnings.extend(proportion_warnings)
            
            return corrections_applied, warnings
            
        except Exception as e:
            print(f"Error applying corrections: {e}")
            return [], [f"Correction error: {str(e)}"]
    
    def _calculate_corrected_value(self, original_value, measurement_name, 
                                  bmi_category, clothing_category, gender):
        """Calculate corrected value using conservative factors."""
        factors = self.correction_factors[measurement_name]
        
        # Start with original value
        corrected_value = original_value
        
        # Apply base correction (small adjustment)
        corrected_value *= factors['base_factor']
        
        # Apply clothing adjustment (small adjustment)
        clothing_factor = factors['clothing_adjustments'].get(clothing_category, 1.0)
        corrected_value *= clothing_factor
        
        # Apply BMI adjustment (small adjustment)
        bmi_factor = factors['bmi_adjustments'].get(bmi_category, 1.0)
        corrected_value *= bmi_factor
        
        # Apply gender adjustment if available (small adjustment)
        if 'gender_adjustments' in factors:
            gender_factor = factors['gender_adjustments'].get(gender, 1.0)
            corrected_value *= gender_factor
        
        return corrected_value
    
    def _check_proportions(self, circumferences, gender):
        """Check proportions and generate warnings for unusual ratios (not forced corrections)."""
        warnings = []
        
        chest = circumferences.get("chest_bust", {}).get("value", 0)
        waist = circumferences.get("waist", {}).get("value", 0)
        hips = circumferences.get("hips", {}).get("value", 0)
        
        if not all([chest, waist, hips]):
            return warnings
        
        # Check waist-to-chest ratio
        waist_chest_ratio = waist / chest
        wc_range = self.proportion_suggestions['waist_to_chest']
        if not (wc_range['min'] <= waist_chest_ratio <= wc_range['max']):
            warnings.append(
                f"Waist-to-chest ratio ({waist_chest_ratio:.2f}) is outside typical range "
                f"({wc_range['min']:.2f}-{wc_range['max']:.2f}). This may be normal for your body type."
            )
        
        # Check hip-to-chest ratio
        hip_chest_ratio = hips / chest
        hc_range = self.proportion_suggestions['hip_to_chest'][gender]
        if not (hc_range['min'] <= hip_chest_ratio <= hc_range['max']):
            warnings.append(
                f"Hip-to-chest ratio ({hip_chest_ratio:.2f}) is outside typical range "
                f"({hc_range['min']:.2f}-{hc_range['max']:.2f}) for {gender}s. "
                f"This may be normal for your body type, especially if you're underweight or have a unique build."
            )
        
        # Check hip-to-waist ratio
        hip_waist_ratio = hips / waist
        hw_range = self.proportion_suggestions['hip_to_waist'][gender]
        if not (hw_range['min'] <= hip_waist_ratio <= hw_range['max']):
            warnings.append(
                f"Hip-to-waist ratio ({hip_waist_ratio:.2f}) is outside typical range "
                f"({hw_range['min']:.2f}-{hw_range['max']:.2f}) for {gender}s."
            )
        
        return warnings


class AdvancedPromptEngine:
    """Creates balanced prompts for body measurement."""

    @staticmethod
    def create_expert_measurement_prompt(height, weight, gender, clothing_desc, camera_distance):
        height_m = float(height) / 100
        bmi = float(weight) / (height_m ** 2)

        if bmi < 18.5: 
            body_category = "underweight"
            proportion_note = "Note: Underweight individuals often have different proportions - chest may be narrower relative to hips."
        elif 18.5 <= bmi < 25: 
            body_category = "normal_weight"
            proportion_note = "Standard proportion guidelines apply."
        elif 25 <= bmi < 30: 
            body_category = "overweight"
            proportion_note = "Note: Overweight individuals may have wider waist relative to chest and hips."
        else: 
            body_category = "obese"
            proportion_note = "Note: Obese individuals typically have wider waist, adjustments for soft tissue expected."

        # Calculate perspective distortion factor
        if float(camera_distance) < 2.0:
            perspective_warning = "CAUTION: Camera distance <2m may cause significant perspective distortion."
            distortion_factor = "high"
        elif float(camera_distance) < 3.0:
            perspective_warning = "Moderate perspective distortion expected."
            distortion_factor = "moderate"
        else:
            perspective_warning = "Optimal camera distance for minimal perspective distortion."
            distortion_factor = "minimal"

        return f"""
ROLE: You are an expert anthropometrist specializing in body measurement extraction from images.

MISSION: Extract accurate body measurements from front and side view images. Be realistic about depth estimation and provide measurements that match what a measuring tape would show.

TECHNICAL SPECIFICATIONS:
• Subject Height: {height} cm (PRIMARY REFERENCE - use for scale calibration)
• Subject Weight: {weight} kg
• BMI: {bmi:.1f} ({body_category})
• Gender: {gender}
• Clothing: {clothing_desc}
• Camera Distance: {camera_distance} meters
• Perspective Distortion Level: {distortion_factor}
• {perspective_warning}

BODY TYPE CONSIDERATIONS:
{proportion_note}

MEASUREMENT PROTOCOL:

**STEP 1: SCALE CALIBRATION**
- Use the known height of {height} cm to establish accurate pixel-to-cm ratio
- Account for perspective distortion based on camera distance
- Verify scale with multiple reference points

**STEP 2: WIDTH MEASUREMENTS (Front View)**
- Shoulder width: Outer edges of deltoids
- Chest width: Widest point at nipple/bust level (typically 4th-5th rib)
- Waist width: Fullest part of torso (usually at or slightly above navel)
- Hip width: Widest point of hips/buttocks (typically at greater trochanter level)

**STEP 3: DEPTH ESTIMATION (Side View)**
- Chest depth: Estimate from side view profile, typically 40-55% of chest width
- Waist depth: Estimate from side view, typically 50-65% of waist width
- Hip depth: Estimate conservatively, typically 25-40% of hip width
  * Be cautious: Hip depth is often overestimated from images
  * Consider that hips are primarily bony structure, not as deep as chest/waist

**STEP 4: CIRCUMFERENCE CALCULATION**
Use standard ellipse approximation: C ≈ π × (width + depth)

For body measurements:
- Chest: π × (chest_width + chest_depth)
- Waist: π × (waist_width + waist_depth)
- Hips: π × (hip_width + hip_depth)

**STEP 5: CLOTHING ADJUSTMENTS**
Apply minimal adjustments based on clothing type:
- Skin-tight/compression: 0 cm
- Fitted: +1 cm
- Regular: +2 cm
- Loose: +3 cm

**STEP 6: REALITY CHECK**
- Verify measurements are physically plausible for the subject's height and weight
- Check that proportions are reasonable for the body type
- For underweight individuals: chest may be smaller relative to hips
- For overweight/obese: waist typically larger relative to chest and hips
- DO NOT force measurements into "ideal" proportions - real bodies vary widely

IMPORTANT GUIDELINES:
1. **Be body-type aware**: Different body types have different proportions
2. **Conservative depth estimates**: When uncertain, underestimate depth slightly
3. **Realistic measurements**: Match what a measuring tape would show
4. **Wide variation is normal**: Don't force measurements into narrow ranges
5. **Height is absolute truth**: Use it to validate all other measurements

SUBJECT PROFILE:
- Height: {height} cm
- Weight: {weight} kg  
- Gender: {gender}
- BMI: {bmi:.1f} ({body_category})
- Clothing: {clothing_desc}
- Camera Distance: {camera_distance} meters

REQUIRED OUTPUT FORMAT:
Provide ONLY a valid JSON object with this structure:

{{
  "analysis_metadata": {{
    "timestamp": "{datetime.datetime.now().isoformat()}",
    "camera_distance_m": {camera_distance},
    "perspective_distortion": "{distortion_factor}",
    "scale_factor_cm_per_pixel": "CALCULATED_VALUE",
    "measurement_accuracy_confidence": "PERCENTAGE",
    "methodology": "conservative_realistic_measurement"
  }},
  "subject_profile": {{
    "height_cm": {height},
    "weight_kg": {weight},
    "gender": "{gender}",
    "bmi": {bmi:.1f},
    "body_category": "{body_category}",
    "clothing_type": "{clothing_desc}"
  }},
  "measurements": {{
    "circumferences_cm": {{
      "chest_bust": {{
        "value": MEASURED_VALUE,
        "visible_width_cm": FRONT_VIEW_WIDTH,
        "estimated_depth_cm": SIDE_VIEW_DEPTH, 
        "confidence": "PERCENTAGE",
        "method": "ellipse_approximation"
      }},
      "waist": {{
        "value": MEASURED_VALUE,
        "visible_width_cm": FULLEST_TORSO_WIDTH,
        "estimated_depth_cm": REALISTIC_DEPTH,
        "confidence": "PERCENTAGE", 
        "method": "ellipse_approximation"
      }},
      "hips": {{
        "value": MEASURED_VALUE,
        "visible_width_cm": HIP_WIDTH,
        "estimated_depth_cm": CONSERVATIVE_DEPTH,
        "confidence": "PERCENTAGE",
        "method": "ellipse_approximation",
        "notes": "Conservative depth estimation applied"
      }}
    }},
    "linear_measurements_cm": {{
      "shoulder_width": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "arm_length": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "leg_length": {{"value": MEASURED, "confidence": "PERCENTAGE"}},
      "neck_circumference": {{"value": ESTIMATED, "confidence": "PERCENTAGE"}}
    }}
  }},
  "quality_assessment": {{
    "image_quality": "excellent/good/fair/poor",
    "pose_accuracy": "excellent/good/fair/poor", 
    "lighting_conditions": "excellent/good/fair/poor",
    "measurement_limitations": ["LIST_ANY_LIMITATIONS"],
    "accuracy_notes": "NOTES_ON_MEASUREMENT_QUALITY"
  }}
}}

CRITICAL REMINDERS:
- Use height as absolute reference for scale
- Be realistic with depth estimates (especially hips)
- Different body types have different proportions - this is normal
- Underweight individuals often have narrower chest relative to hips
- Provide measurements that would match a real measuring tape
"""


# Global instance
measurement_corrector = MeasurementCorrector()

def encode_image_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def validate_image(file):
    """Validate uploaded image file."""
    if not file or file.filename == '':
        return False, "No file provided"
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return False, "Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > 16 * 1024 * 1024:  # 16MB
        return False, "File too large. Maximum size: 16MB"
    
    # Validate it's a real image
    try:
        Image.open(file).verify()
        file.seek(0)
        return True, "Valid image"
    except Exception:
        return False, "Invalid or corrupted image file"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Body Measurement API',
        'version': '2.0.0',
        'timestamp': datetime.datetime.now().isoformat(),
        'features': 'Conservative body-type-aware corrections'
    })

@app.route('/analyze', methods=['POST'])
def analyze_measurements():
    """
    Main API endpoint for body measurement analysis.
    Accepts multipart form data with images and measurement parameters.
    """
    try:
        # Validate request content type
        if 'multipart/form-data' not in request.content_type:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be multipart/form-data',
                'error_code': 'INVALID_CONTENT_TYPE'
            }), 400

        # Extract and validate images
        if 'front_image' not in request.files or 'side_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both front_image and side_image are required',
                'error_code': 'MISSING_IMAGES'
            }), 400

        front_file = request.files['front_image']
        side_file = request.files['side_image']

        # Validate images
        is_valid, message = validate_image(front_file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Front image validation failed: {message}',
                'error_code': 'INVALID_FRONT_IMAGE'
            }), 400

        is_valid, message = validate_image(side_file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Side image validation failed: {message}',
                'error_code': 'INVALID_SIDE_IMAGE'
            }), 400

        # Extract and validate form parameters
        required_params = ['height', 'weight', 'gender', 'clothing', 'camera_distance', 'api_key']
        params = {}
        
        for param in required_params:
            value = request.form.get(param)
            if not value:
                return jsonify({
                    'success': False,
                    'error': f'Missing required parameter: {param}',
                    'error_code': 'MISSING_PARAMETER'
                }), 400
            params[param] = value

        # Validate numeric parameters
        try:
            height = float(params['height'])
            weight = float(params['weight'])
            camera_distance = float(params['camera_distance'])
            
            if not (100 <= height <= 250):
                return jsonify({
                    'success': False,
                    'error': 'Height must be between 100-250 cm',
                    'error_code': 'INVALID_HEIGHT'
                }), 400
                
            if not (30 <= weight <= 300):
                return jsonify({
                    'success': False,
                    'error': 'Weight must be between 30-300 kg',
                    'error_code': 'INVALID_WEIGHT'
                }), 400
                
            if not (0.5 <= camera_distance <= 10):
                return jsonify({
                    'success': False,
                    'error': 'Camera distance must be between 0.5-10 meters',
                    'error_code': 'INVALID_CAMERA_DISTANCE'
                }), 400
                
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid numeric values for height, weight, or camera_distance',
                'error_code': 'INVALID_NUMERIC_VALUES'
            }), 400

        # Validate gender
        if params['gender'].lower() not in ['male', 'female']:
            return jsonify({
                'success': False,
                'error': 'Gender must be either "male" or "female"',
                'error_code': 'INVALID_GENDER'
            }), 400

        # Save uploaded files temporarily
        timestamp = int(time.time())
        front_filename = secure_filename(f"front_{timestamp}_{front_file.filename}")
        side_filename = secure_filename(f"side_{timestamp}_{side_file.filename}")
        
        front_path = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        side_path = os.path.join(app.config['UPLOAD_FOLDER'], side_filename)
        
        front_file.save(front_path)
        side_file.save(side_path)

        try:
            # Create measurement prompt
            prompt = AdvancedPromptEngine.create_expert_measurement_prompt(
                params['height'],
                params['weight'],
                params['gender'],
                params['clothing'],
                params['camera_distance']
            )

            # Encode images to base64
            front_b64 = encode_image_base64(front_path)
            side_b64 = encode_image_base64(side_path)

            # Prepare API request to Groq
            headers = {
                "Authorization": f"Bearer {params['api_key']}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{front_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{side_b64}"}}
                    ]
                }],
                "max_tokens": 8192,
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }

            # Make API request
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )

            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'error': f'AI API error: {response.text}',
                    'error_code': 'AI_API_ERROR',
                    'status_code': response.status_code
                }), 500

            # Parse AI response
            try:
                result_json = response.json()['choices'][0]['message']['content']
                measurements = json.loads(result_json)
            except (KeyError, json.JSONDecodeError) as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to parse AI response: {str(e)}',
                    'error_code': 'AI_RESPONSE_PARSE_ERROR'
                }), 500

            # Apply conservative corrections if requested
            apply_corrections = request.form.get('apply_corrections', 'true').lower() == 'true'
            corrections_log = []
            warnings = []
            
            if apply_corrections:
                subject_profile = measurements.get("subject_profile", {})
                corrections_log, warnings = measurement_corrector.apply_corrections(
                    measurements, subject_profile
                )
            
            # Add correction information to response
            measurements.setdefault("quality_assessment", {})
            if corrections_log or warnings:
                measurements["quality_assessment"]["correction_info"] = {
                    "method": "conservative_body_type_aware_correction",
                    "corrections_applied": corrections_log if corrections_log else ["No significant corrections needed"],
                    "warnings": warnings if warnings else ["All proportions within normal ranges"],
                    "timestamp": datetime.datetime.now().isoformat()
                }

            # Success response
            return jsonify({
                'success': True,
                'data': measurements,
                'processing_info': {
                    'corrections_applied': apply_corrections,
                    'correction_count': len(corrections_log),
                    'warnings_count': len(warnings),
                    'api_provider': 'groq_llama',
                    'api_version': '2.0.0',
                    'processing_time': datetime.datetime.now().isoformat()
                }
            })

        finally:
            # Clean up uploaded files
            try:
                os.remove(front_path)
                os.remove(side_path)
            except OSError:
                pass

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_SERVER_ERROR'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB per file.',
        'error_code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'error_code': 'NOT_FOUND'
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'error_code': 'METHOD_NOT_ALLOWED'
    }), 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
