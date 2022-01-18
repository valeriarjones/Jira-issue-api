from datetime import datetime
import flask
from flask_sqlalchemy import SQLAlchemy
from .regression_model import model


db = SQLAlchemy()

def create_app():
    from app.predictions.blueprints import blueprint
    app = flask.Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///issues_preds.db"
    app.register_blueprint(blueprint)

    db.init_app(app)

    data = model.run()
    dtype={"issue": db.String(50),
           "current_status": db.String(50),
           "prediction_resolution_date": db.DateTime}

    db_engine = db.create_engine("sqlite:///app/issues_preds.db", {})

    data.to_sql(name="issue", con=db_engine, if_exists="replace", index=False, dtype=dtype)

    return app
