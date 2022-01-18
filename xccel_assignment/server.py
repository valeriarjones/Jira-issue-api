"""
server.py - A minimal Flask application

Usage: Run 'export FLASK_APP=server' followed by 'flask run' ('export' if unix shell, otherwise use 'set')

Project: Building an API with Python & Flask to access JIRA issues resolution predictions from a sqlite db

Author: Valeria Jones
"""

from app import create_app
from flask_sqlalchemy import SQLAlchemy

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0", port=5000)