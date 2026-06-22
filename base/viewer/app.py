"""Bud-Analysis viewer — Flask app.

Interactive predicted-vs-true explorer for a trained run: filter by view / fork,
choose the aggregation (per view, per fork = MIL mean, per flower), and click a
point to inspect that flower's views. Run from this directory:

    python app.py            # http://127.0.0.1:5000

Needs the core env (torch/numpy/pandas/Flask) — installed with base.
"""

from flask import Flask

from routes.api import api_bp
from routes.web import web_bp

app = Flask(__name__)
app.register_blueprint(web_bp)
app.register_blueprint(api_bp)


if __name__ == "__main__":
    app.run(debug=True)
