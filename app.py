from flask import Flask, render_template, request, jsonify
import os
import base64
from PIL import Image
import io
from groq import Groq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def validate_image(image_data):
    """Validate the image data and return base64 string"""
    try:
        # Check if it's a base64 string
        if image_data.startswith('data:image'):
            # Validate by trying to decode
            Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1])))
            return image_data
        else:
            raise ValueError("Invalid image format")
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def generate_prompt_from_image(image_data):
    """Generate prompt using Groq API with image description"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail including style, composition, colors, and key elements. Structure your response as: 'Subject: [main subject], Style: [art style], Colors: [color palette], Details: [key details]'"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating prompt: {str(e)}")
        raise Exception("Failed to generate prompt. Please try another image.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if 'image-data' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Validate and process image
        image_data = validate_image(data['image-data'])
        
        # Generate prompt
        prompt = generate_prompt_from_image(image_data)
        return jsonify({"prompt": prompt})
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)