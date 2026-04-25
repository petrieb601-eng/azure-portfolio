from flask import Flask, render_template, request, jsonify
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential as VisionKeyCredential
from openai import AzureOpenAI
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from werkzeug.utils import secure_filename
import base64

load_dotenv()

app = Flask(__name__)

# File encryption configuration
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'zip'}

# Fake data for now (we'll connect to database later)
fake_projects = [
    {
        'id': 1,
        'name': 'Advanced Game Design',
        'description': 'Developed, debugged, and implemented mechanics into a game while maintaining original functionality. Created player movement, collision detection, and scoring systems.',
        'technologies': 'Java, Object-Oriented Programming, Game Development',
        'github_url': 'https://github.com/yourusername/game-project'
    },
    {
        'id': 2,
        'name': 'POS Development',
        'description': 'Assisted in the implementation and development of a Point of Service system for a local establishment. Consulted staff to ensure the system was user-friendly and flexible.',
        'technologies': 'Python, SQL, Database Design, UI/UX',
        'github_url': 'https://github.com/yourusername/pos-system'
    },
    {
        'id': 3,
        'name': 'Azure Portfolio Website',
        'description': 'Deployed full-stack web application on Azure App Service with automated CI/CD from GitHub. Integrated Azure SQL Database for dynamic content management.',
        'technologies': 'Python, Flask, Azure, SQL, HTML/CSS',
        'github_url': 'https://github.com/yourusername/azure-portfolio'
    }
]

# Azure AI setup
def authenticate_client():
    key = os.getenv('AZURE_LANGUAGE_KEY')
    endpoint = os.getenv('AZURE_LANGUAGE_ENDPOINT')
    if key and endpoint:
        return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return None

def authenticate_vision_client():
    key = os.getenv('AZURE_VISION_KEY')
    endpoint = os.getenv('AZURE_VISION_ENDPOINT')
    if key and endpoint:
        return ImageAnalysisClient(endpoint=endpoint, credential=VisionKeyCredential(key))
    return None

def get_openai_client():
    api_key = os.getenv('AZURE_OPENAI_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if api_key and endpoint:
        return AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=endpoint,
            http_client=None
        )
    return None

def load_knowledge_base():
    try:
        with open('knowledge_base.txt', 'r') as f:
            return f.read()
    except:
        return "No knowledge base available."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_key_from_password(password, salt):
    """Generate encryption key from password"""
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/projects')
def projects():
    return render_template('projects.html', projects=fake_projects)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    client = authenticate_client()
    if not client:
        return jsonify({'error': 'AI service not configured'}), 500
    
    text = request.json.get('text', '')
    
    try:
        response = client.analyze_sentiment(documents=[text])[0]
        
        return jsonify({
            'sentiment': response.sentiment,
            'positive': response.confidence_scores.positive,
            'neutral': response.confidence_scores.neutral,
            'negative': response.confidence_scores.negative
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image-analysis')
def image_analysis():
    return render_template('image_analysis.html')

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    client = authenticate_vision_client()
    if not client:
        return jsonify({'error': 'Vision service not configured'}), 500
    
    try:
        # Get image URL from request
        image_url = request.json.get('image_url', '')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # Analyze image - using only universally supported features
        result = client.analyze(
            image_url=image_url,
            visual_features=[
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.READ
            ]
        )
        
        # Create a description from tags
        top_tags = [tag.name for tag in (result.tags.list[:3] if result.tags else [])]
        generated_caption = f"An image featuring {', '.join(top_tags)}" if top_tags else "Image analyzed"
        
        # Extract results with safer handling
        response_data = {
            'caption': generated_caption,
            'confidence': result.tags.list[0].confidence if result.tags and result.tags.list else 0,
            'objects': [{'name': obj.tags[0].name if obj.tags else 'Unknown', 
                        'confidence': obj.tags[0].confidence if obj.tags else 0} 
                       for obj in (result.objects.list if result.objects else [])],
            'tags': [{'name': tag.name, 'confidence': tag.confidence} 
                    for tag in (result.tags.list[:10] if result.tags else [])],
            'text': [line.text for block in (result.read.blocks if result.read else []) 
                    for line in block.lines]
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    client = get_openai_client()
    if not client:
        return jsonify({'error': 'OpenAI service not configured'}), 500
    
    try:
        user_message = request.json.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Load knowledge base
        knowledge = load_knowledge_base()
        
        # Create system message with RAG context
        system_message = f"""You are a helpful assistant that answers questions about Bryan Petrie based on the following information.
        
KNOWLEDGE BASE:
{knowledge}

Instructions:
- Only answer questions based on the information provided above
- If the information isn't in the knowledge base, say "I don't have that information about Bryan"
- Be friendly and conversational
- Keep answers concise but informative
- You can elaborate on the information provided, but don't make things up"""

        # Call Azure OpenAI
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'chat-model')
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content
        
        return jsonify({'response': assistant_message})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/encryption')
def encryption():
    return render_template('encryption.html')

@app.route('/encrypt-file', methods=['POST'])
def encrypt_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        password = request.form.get('password', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        if file and allowed_file(file.filename):
            # Read file data
            file_data = file.read()
            
            # Generate salt and key
            salt = os.urandom(16)
            key = generate_key_from_password(password, salt)
            
            # Encrypt the file
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(file_data)
            
            # Combine salt + encrypted data
            final_data = salt + encrypted_data
            
            # Encode to base64 for JSON response
            encrypted_base64 = base64.b64encode(final_data).decode('utf-8')
            
            return jsonify({
                'success': True,
                'encrypted_data': encrypted_base64,
                'filename': secure_filename(file.filename) + '.encrypted'
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/decrypt-file', methods=['POST'])
def decrypt_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        password = request.form.get('password', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        # Read encrypted file
        encrypted_file_data = file.read()
        
        # Extract salt (first 16 bytes) and encrypted data
        salt = encrypted_file_data[:16]
        encrypted_data = encrypted_file_data[16:]
        
        # Generate key from password
        key = generate_key_from_password(password, salt)
        
        # Decrypt the file
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_data)
        
        # Encode to base64 for JSON response
        decrypted_base64 = base64.b64encode(decrypted_data).decode('utf-8')
        
        # Remove .encrypted extension if present
        original_filename = file.filename.replace('.encrypted', '')
        
        return jsonify({
            'success': True,
            'decrypted_data': decrypted_base64,
            'filename': secure_filename(original_filename)
        })
        
    except Exception as e:
        return jsonify({'error': 'Decryption failed. Wrong password or corrupted file.'}), 500

if __name__ == '__main__':
    app.run(debug=True)