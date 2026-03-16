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