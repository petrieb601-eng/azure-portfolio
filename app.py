from flask import Flask, render_template
import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Database connection function
def get_db_connection():
    try:
        connection_string = os.getenv('AZURE_SQL_CONNECTION_STRING')
        conn = pyodbc.connect(connection_string)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/projects')
def projects():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, technologies, github_url FROM projects ORDER BY id DESC')
        projects_data = cursor.fetchall()
        conn.close()
        return render_template('projects.html', projects=projects_data)
    else:
        return render_template('projects.html', projects=[], error="Database connection failed")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)