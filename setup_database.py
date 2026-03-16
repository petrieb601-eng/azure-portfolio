import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

# Get connection string
conn_str = os.getenv('AZURE_SQL_CONNECTION_STRING')

print("Attempting to connect to database...")

try:
    # Connect to database
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    print("✅ Connected successfully!")
    
    # Create table
    print("\nCreating projects table...")
    cursor.execute('''
        CREATE TABLE projects (
            id INT PRIMARY KEY IDENTITY(1,1),
            name VARCHAR(100) NOT NULL,
            description TEXT NOT NULL,
            technologies VARCHAR(200),
            github_url VARCHAR(200),
            created_date DATETIME DEFAULT GETDATE()
        )
    ''')
    conn.commit()
    print("✅ Table created successfully!")
    
    # Insert projects
    print("\nInserting projects...")
    projects = [
        ('Advanced Game Design', 
         'Developed, debugged, and implemented mechanics into a game while maintaining original functionality. Created player movement, collision detection, and scoring systems.', 
         'Java, Object-Oriented Programming, Game Development', 
         'https://github.com/yourusername/game-project'),
        
        ('POS Development', 
         'Assisted in the implementation and development of a Point of Service system for a local establishment. Consulted staff to ensure the system was user-friendly and flexible.', 
         'Python, SQL, Database Design, UI/UX', 
         'https://github.com/yourusername/pos-system'),
        
        ('Azure Portfolio Website', 
         'Deployed full-stack web application on Azure App Service with automated CI/CD from GitHub. Integrated Azure SQL Database for dynamic content management.', 
         'Python, Flask, Azure, SQL, HTML/CSS', 
         'https://github.com/yourusername/azure-portfolio')
    ]
    
    for project in projects:
        cursor.execute('''
            INSERT INTO projects (name, description, technologies, github_url)
            VALUES (?, ?, ?, ?)
        ''', project)
    
    conn.commit()
    print("✅ Projects inserted successfully!")
    
    # Verify data
    print("\nVerifying data...")
    cursor.execute("SELECT id, name FROM projects")
    rows = cursor.fetchall()
    print(f"\n✅ Found {len(rows)} projects:")
    for row in rows:
        print(f"  - {row.id}: {row.name}")
    
    cursor.close()
    conn.close()
    print("\n✅ Database setup complete!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")