import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    make_response,
    session,
    redirect,
    url_for,
    jsonify,
)
import argparse


app = Flask(__name__, root_path="./", template_folder="./templates")


@app.route("/")
def index():
    page = render_template("index.html")
    return page

@app.route("/search", methods=["POST"])
def search_similar_weather_date():
    pass



if __name__=="__main__":
    import os
    print(os.getcwd())
    app.debug=True
    app.run()