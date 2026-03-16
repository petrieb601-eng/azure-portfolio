from flask import Flask, render_template

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/projects')
def projects():
    return render_template('projects.html', projects=fake_projects)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Azure AI setup
def authenticate_client():
    key = os.getenv('AZURE_LANGUAGE_KEY')
    endpoint = os.getenv('AZURE_LANGUAGE_ENDPOINT')
    if key and endpoint:
        return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return None

# Your existing routes...
# (keep all your current routes)

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