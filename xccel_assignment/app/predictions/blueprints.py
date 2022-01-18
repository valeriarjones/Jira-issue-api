"""
Module containing the API GET endpoint definitions.
The endpoints that have defined are:
- localhost:PORT/issues
- localhost:PORT/issues/resolve-fake
- localhost:PORT/issues/<string:uid>/resolve-prediction
- localhost:PORT/release/<dtime>/resolved-since-now
"""
import flask
from app import db
import datetime
from dateutil import parser
from models import Issue


blueprint = flask.Blueprint("blueprint", __name__, url_prefix="/")


# Define all issues endpoint
@blueprint.route("issues", methods=["GET"])
def get_predictions():
    predictions = Issue.query.all()
    serialised = {
        "predictions": [prediction.serialised() for prediction in predictions]
    }
    return flask.jsonify(serialised)


# Define fake prediction for fake issue
resolve_fake = [
        {
               'issue' : 'AVRO-9999',
               'predicted_resolution_date' : '1970-01-01T00:00:00.000+0000'
        }
               ]


# Define fake resolution endpoint
@blueprint.route("issues/resolve-fake", methods=["GET"])
def get_fake():
    serialised = {
        "resolve_fake": resolve_fake
    }
    return flask.jsonify(serialised)


# Define issue prediction endpoint
@blueprint.route("issues/<string:uid>/resolve-prediction", methods=["GET"])
def get_prediction(uid):
    prediction = Issue.query.get_or_404(uid)

    return flask.jsonify(prediction.serialised())


# Define release assistance endpoint
@blueprint.route("release/<dtime>/resolved-since-now", methods=["GET"])
def get_release_assist(dtime):
    now = datetime.datetime.now()
    dtime = parser.parse(dtime)
    issues = Issue.query.filter(Issue.prediction_resolution_date < dtime).filter(Issue.prediction_resolution_date >= now)
    if issues:
        serialised = {
            "now": datetime.datetime.strftime(now, '%Y-%m-%dT%H:%M:%SZ'),
            "issues": [issue.serialised() for issue in issues]
        }
    else:
        serialised = {
            "now": datetime.datetime.strftime(now, '%Y-%m-%dT%H:%M:%SZ'),
            "issues": []
        }
    return flask.jsonify(serialised)
