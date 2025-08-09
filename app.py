from flask import Flask, request, jsonify
import base64
import requests
import json
import os
import datetime
import math

# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class MeasurementCorrector:
    """Handles measurement corrections and validation with research-based methods."""
    
    def __init__(self):
        # Research-based correction factors
        self.correction_factors = {
            'chest_bust': {
                'base_factor': 1.05,
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 1.02,
                    'regular': 1.04,
                    'loose': 1.06
                },
                'bmi_adjustments': {
                    'underweight': 1.00,
                    'normal': 1.02,
                    'overweight': 1.03,
                    'obese': 1.05
                }
            },
            'waist': {
                'base_factor': 1.12,
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 1.03,
                    'regular': 1.06,
                    'loose': 1.08
                },
                'bmi_adjustments': {
                    'underweight': 1.02,
                    'normal': 1.04,
                    'overweight': 1.06,
                    'obese': 1.08
                }
            },
            'hips': {
                'base_factor': 0.55,
                'clothing_adjustments': {
                    'skin-tight': 1.00,
                    'fitted': 0.95,
                    'regular': 0.90,
                    'loose': 0.85
                },
                'bmi_adjustments': {
                    'underweight': 0.80,
                    'normal': 0.85,
                    'overweight': 0.90,
                    'obese': 0.95
                },
                'gender_adjustments': {
                    'male': 0.65,
                    'female': 0.75
                }
            }
        }
        
        # Proportion validation ranges
        self.proportion_ranges = {
            'waist_to_chest': {'min': 0.75, 'max': 1.20},
            'hip_to_chest': {
                'male': {'min': 0.70, 'max': 0.88},
                'female': {'min': 0.80, 'max': 0.98}
            },
            'hip_to_waist': {
                'male': {'min': 0.85, 'max': 1.00},
                'female': {'min': 0.90, 'max': 1.10}
            }
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
        """Apply research-based corrections to measurements."""
        try:
            corrections_applied = []
            circumferences = measurements_data.get("measurements", {}).get("circumferences_cm", {})
            
            # Get subject characteristics
            height = float(subject_profile.get("height_cm", 170))
            weight = float(subject_profile.get("weight_kg", 70))
            gender = subject_profile.get("gender", "Male").lower()
            clothing = subject_profile.get("clothing_type", "fitted")
            
            bmi_category, bmi_value = self.get_bmi_category(height, weight)
            clothing_category = self.get_clothing_category(clothing)
            
            # Apply corrections to each measurement
            for measurement_name, measurement_data in circumferences.items():
                if measurement_name in self.correction_factors and "value" in measurement_data:
                    original_value = float(measurement_data["value"])
                    corrected_value = self._calculate_corrected_value(
                        original_value, measurement_name, bmi_category, 
                        clothing_category, gender
                    )
                    
                    if abs(corrected_value - original_value) > 0.5:
                        measurement_data["value"] = round(corrected_value, 1)
                        measurement_data["correction_applied"] = True
                        measurement_data["correction_notes"] = f"Research-based correction applied"
                        corrections_applied.append(
                            f"{measurement_name}: {original_value:.1f} → {corrected_value:.1f} cm"
                        )
            
            # Validate proportions with gender-specific limits
            proportion_fixes = self._validate_proportions(circumferences, gender)
            corrections_applied.extend(proportion_fixes)
            
            return corrections_applied
            
        except Exception as e:
            print(f"Error applying corrections: {e}")
            return []
    
    def _calculate_corrected_value(self, original_value, measurement_name, 
                                 bmi_category, clothing_category, gender):
        """Calculate corrected value using research-based factors."""
        factors = self.correction_factors[measurement_name]
        
        # Start with base correction
        corrected_value = original_value * factors['base_factor']
        
        # Apply clothing adjustment
        clothing_factor = factors['clothing_adjustments'].get(clothing_category, 1.0)
        corrected_value *= clothing_factor
        
        # Apply BMI adjustment
        bmi_factor = factors['bmi_adjustments'].get(bmi_category, 1.0)
        corrected_value *= bmi_factor
        
        # Apply gender adjustment if available
        if 'gender_adjustments' in factors:
            gender_factor = factors['gender_adjustments'].get(gender, 1.0)
            corrected_value *= gender_factor
        
        return corrected_value
    
    def _validate_proportions(self, circumferences, gender):
        """Validate and fix unrealistic proportions with gender-specific limits."""
        fixes = []
        
        chest = circumferences.get("chest_bust", {}).get("value", 0)
        waist = circumferences.get("waist", {}).get("value", 0)
        hips = circumferences.get("hips", {}).get("value", 0)
        
        if not all([chest, waist, hips]):
            return fixes
        
        # Check waist-to-chest ratio
        waist_chest_ratio = waist / chest
        if waist_chest_ratio < self.proportion_ranges['waist_to_chest']['min']:
            corrected_waist = chest * self.proportion_ranges['waist_to_chest']['min']
            circumferences["waist"]["value"] = round(corrected_waist, 1)
            fixes.append(f"Waist proportion corrected: {waist:.1f} → {corrected_waist:.1f} cm")
            waist = corrected_waist
        
        # Check hip-to-chest ratio with gender-specific limits
        hip_chest_ratio = hips / chest
        gender_hip_limits = self.proportion_ranges['hip_to_chest'][gender]
        
        if hip_chest_ratio > gender_hip_limits['max']:
            corrected_hips = chest * gender_hip_limits['max']
            circumferences["hips"]["value"] = round(corrected_hips, 1)
            fixes.append(f"Hip-to-chest proportion corrected: {hips:.1f} → {corrected_hips:.1f} cm")
            hips = corrected_hips
        
        # Check hip-to-waist ratio with gender-specific limits
        hip_waist_ratio = hips / waist
        gender_waist_limits = self.proportion_ranges['hip_to_waist'][gender]
        
        if hip_waist_ratio > gender_waist_limits['max']:
            corrected_hips = waist * gender_waist_limits['max']
            if corrected_hips < circumferences["hips"]["value"]:
                circumferences["hips"]["value"] = round(corrected_hips, 1)
                fixes.append(f"Hip-to-waist proportion corrected: {hips:.1f} → {corrected_hips:.1f} cm")
        
        return fixes


class AdvancedPromptEngine:
    """Creates research-based prompts for body measurement."""

    @staticmethod
    def create_expert_measurement_prompt(height, weight, gender, clothing_desc, camera_distance):
        height_m = float(height) / 100
        bmi = float(weight) / (height_m ** 2)

        if bmi < 18.5: body_category = "underweight"
        elif 18.5 <= bmi < 25: body_category = "normal_weight"
        elif 25 <= bmi < 30: body_category = "overweight"
        else: body_category = "obese"

        if float(camera_distance) < 2.0:
            perspective_warning = "CRITICAL: Camera distance <2m may cause significant perspective distortion."
            distortion_factor = "high"
        elif float(camera_distance) < 3.0:
            perspective_warning = "Moderate perspective distortion expected."
            distortion_factor = "moderate"
        else:
            perspective_warning = "Optimal camera distance for minimal perspective distortion."
            distortion_factor = "minimal"

        return f"""You are an expert anthropometrist. Extract precise body measurements from images with conservative hip estimation.

SUBJECT: Height {height}cm, Weight {weight}kg, {gender}, BMI {bmi:.1f}, Camera {camera_distance}m

CRITICAL: Hip measurements are commonly overestimated by AI. Use conservative calculations.

REQUIRED JSON OUTPUT:
{{
  "analysis_metadata": {{
    "timestamp": "{datetime.datetime.now().isoformat()}",
    "camera_distance_m": {camera_distance},
    "perspective_distortion": "{distortion_factor}",
    "methodology": "conservative_measurement"
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
      "chest_bust": {{"value": "MEASURED_VALUE", "confidence": "PERCENTAGE"}},
      "waist": {{"value": "MEASURED_VALUE", "confidence": "PERCENTAGE"}},
      "hips": {{"value": "CONSERVATIVE_MEASURED_VALUE", "confidence": "PERCENTAGE"}}
    }},
    "linear_measurements_cm": {{
      "shoulder_width": {{"value": "MEASURED", "confidence": "PERCENTAGE"}},
      "arm_length": {{"value": "MEASURED", "confidence": "PERCENTAGE"}},
      "leg_length": {{"value": "MEASURED", "confidence": "PERCENTAGE"}},
      "neck_circumference": {{"value": "ESTIMATED", "confidence": "PERCENTAGE"}}
    }}
  }},
  "quality_assessment": {{
    "image_quality": "excellent/good/fair/poor",
    "pose_accuracy": "excellent/good/fair/poor",
    "lighting_conditions": "excellent/good/fair/poor",
    "measurement_limitations": ["LIST_LIMITATIONS"],
    "accuracy_notes": "ASSESSMENT_NOTES"
  }}
}}"""

# Global instances
measurement_corrector = MeasurementCorrector()

def encode_image_base64(image_data):
    """Encode image data to base64."""
    return base64.b64encode(image_data).decode('utf-8')

def generate_measurement_summary(measurements):
    """Generate a clean summary of measurements."""
    try:
        circumferences = measurements.get("measurements", {}).get("circumferences_cm", {})
        analysis_meta = measurements.get("analysis_metadata", {})
        subject_profile = measurements.get("subject_profile", {})
        
        summary = {
            "subject_info": {
                "gender": subject_profile.get('gender', 'N/A'),
                "height_cm": subject_profile.get('height_cm', 'N/A'),
                "weight_kg": subject_profile.get('weight_kg', 'N/A'),
                "bmi": subject_profile.get('bmi', 'N/A'),
                "body_category": subject_profile.get('body_category', 'N/A'),
                "camera_distance_m": analysis_meta.get('camera_distance_m', 'N/A'),
                "distortion_level": analysis_meta.get('perspective_distortion', 'N/A')
            },
            "measurements_summary": {},
            "corrections_applied": []
        }
        
        for measurement_name, data in circumferences.items():
            if "value" in data:
                display_name = measurement_name.replace('_', ' ').title()
                summary["measurements_summary"][measurement_name] = {
                    "name": display_name,
                    "value": data['value'],
                    "confidence": data.get('confidence', 'N/A'),
                    "corrected": data.get('correction_applied', False)
                }
        
        # Check for applied corrections
        corrections = measurements.get("quality_assessment", {}).get("research_corrections", {})
        if corrections:
            applied_corrections = corrections.get("corrections_applied", [])
            summary["corrections_applied"] = applied_corrections
        
        return summary
        
    except Exception as e:
        return {"error": f"Error generating summary: {e}"}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Body Measurement API",
        "platform": "Vercel Serverless"
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        "message": "Body Measurement API is running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze"
        },
        "status": "ready"
    })

@app.route('/analyze', methods=['POST'])
def analyze_body_measurements():
    """Main endpoint for body measurement analysis."""
    try:
        # Check if images are present
        if 'front_image' not in request.files or 'side_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both front_image and side_image are required'
            }), 400
        
        front_file = request.files['front_image']
        side_file = request.files['side_image']
        
        if front_file.filename == '' or side_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Get form parameters
        height = request.form.get('height')
        weight = request.form.get('weight')
        gender = request.form.get('gender', 'Male')
        clothing = request.form.get('clothing', 'fitted')
        camera_distance = request.form.get('camera_distance', '2.5')
        api_key = request.form.get('api_key')
        apply_corrections = request.form.get('apply_corrections', 'true').lower() == 'true'
        
        # Validate required parameters
        if not all([height, weight, api_key]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: height, weight, api_key'
            }), 400
        
        # Validate numeric inputs
        try:
            height_val = float(height)
            weight_val = float(weight)
            camera_distance_val = float(camera_distance)
            
            if not (100 <= height_val <= 250):
                return jsonify({
                    'success': False,
                    'error': 'Height must be between 100-250 cm'
                }), 400
            if not (30 <= weight_val <= 300):
                return jsonify({
                    'success': False,
                    'error': 'Weight must be between 30-300 kg'
                }), 400
            if not (0.5 <= camera_distance_val <= 10):
                return jsonify({
                    'success': False,
                    'error': 'Camera distance must be between 0.5-10 meters'
                }), 400
                
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid numeric values for height, weight, or camera_distance'
            }), 400
        
        # Read and encode images
        front_image_data = front_file.read()
        side_image_data = side_file.read()
        
        front_b64 = encode_image_base64(front_image_data)
        side_b64 = encode_image_base64(side_image_data)
        
        # Create prompt
        prompt = AdvancedPromptEngine.create_expert_measurement_prompt(
            height, weight, gender, clothing, camera_distance
        )
        
        # Prepare API request to Groq
        headers = {
            "Authorization": f"Bearer {api_key}",
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
            timeout=20
        )
        
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': f"AI API Error {response.status_code}: {response.text}"
            }), 500
        
        # Parse AI response
        try:
            result_json = response.json()['choices'][0]['message']['content']
            measurements = json.loads(result_json)
        except (KeyError, json.JSONDecodeError) as e:
            return jsonify({
                'success': False,
                'error': f'Failed to parse AI response: {str(e)}'
            }), 500
        
        # Apply corrections if requested
        corrections_applied = []
        if apply_corrections:
            subject_profile = measurements.get("subject_profile", {})
            corrections_applied = measurement_corrector.apply_corrections(measurements, subject_profile)
            
            if corrections_applied:
                measurements.setdefault("quality_assessment", {})["research_corrections"] = {
                    "method": "research_based_measurement_correction",
                    "corrections_applied": corrections_applied,
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        # Generate summary
        summary = generate_measurement_summary(measurements)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.datetime.now().isoformat(),
            'measurements': measurements,
            'summary': summary,
            'corrections_applied': corrections_applied,
            'corrections_count': len(corrections_applied)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB per image.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# CRITICAL: This is the correct way to export for Vercel
# Do not modify this section
def handler(request):
    return app(request.environ, lambda *args: None)

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
