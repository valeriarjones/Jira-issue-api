### MLE assessment - Xccelerated
#### Jira issue resolution prediction and release planning assistance 
A REST-ish API that takes a Jira issue key and returns the prediction 
date for the issue to reach the status Resolved.
Two REST endpoints have been implemented:
- GET URL/issues/{Issue-key}/resolve-prediction
- GET URL/release/{date}/resolved-since-now

Project structure:

```
xccel_assignment/
|
|-app/
| |-predictions/
| | |-__init__.py
| | |-blueprints.py
| |
| |-regression_model/
| | |-__init__.py
| | |-model.py
| |-__init__.py
|
|-data/
| |-avro-daycounts.csv
| |-avro-issues.csv
| |-avro-transitions.csv
| |-config.yaml
|
|-models.py
|-server.py
```
When the service is started, before creating the app, the model to 
obtain the prediction date for the issues is generated. The results 
are then saved to a local sqlite database.

The script to generate the model to predict when an issue will be resolved is
implemented in xccel_assignment/app/regression_model.

The dependencies are:
- python 3.9
- Flask
- Flask-SQLAlchemy
- numpy
- pandas
- scikit-learn
- pyyaml

**Usage**: Change to xccel_assignment/ directory. Run first 'export FLASK_APP=server', and then 'flask run' from terminal 
('export' if unix shell, otherwise use 'set'). Open browser and copy URL and paste it + endpoint_path into the browser.

**Change datasets**: Open data/config.yaml and change name and 
path (to data) fields for each dataset with desired ones. Restart service.
