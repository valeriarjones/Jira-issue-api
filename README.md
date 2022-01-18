###MLE assessment - Xccelerated
#### Jira issue resolution prediction and release planning assistance 
A REST-ish API that takes a Jira issue key and returns the prediction 
date for the issue to reach the status Resolved.
Two REST endpoints have been implemented:
- GET URL/issues/{Issue-key}/resolve-prediction
- GET URL/release/{date}/resolved-since-now

When the service is started, before creating the app, the model to 
obtain the prediction date for the issues is generated. The results 
are then saved to a local sqlite database.

The dependencies are:
- python
- Flask
- Flask-SQLAlchemy
- numpy
- pandas
- scikit-learn
- pyyaml

**Usage**: Run first 'export FLASK_APP=server', and then 'flask run' from terminal 
('export' if unix shell, otherwise use 'set'). Open browser and copy URL. To reach 
the endpoints, add the endpoints paths.

**Change datasets**: Open data/config.yaml and change name and 
path (to data) fields for each dataset with desired ones. Restart service.