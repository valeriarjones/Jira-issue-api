"""
Module containing the models for the sqlite db.
"""

from app import db
from datetime import datetime


class Issue(db.Model):
    issue = db.Column(db.String(50), primary_key=True)
    current_status = db.Column(db.String(50), nullable=False)
    prediction_resolution_date = db.Column(db.DateTime, nullable=False)

    def serialised(self):
        json = {
            "issue": self.issue,
            "current_status": self.current_status,
            "prediction_resolution_date": datetime.strftime(self.prediction_resolution_date, '%Y-%m-%dT%H:%M:%SZ')
        }
        return json