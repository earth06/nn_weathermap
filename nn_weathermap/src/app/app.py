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
import xarray as xr 
from src.ai.index_search import IndexSearch

DS = xr.open_dataset("./data/era5_asia_12h_199001_202308_0900_2100JST.nc")
INDEX = IndexSearch()
INDEX.load_weather_index()

app = Flask(__name__, root_path="./", template_folder="./templates")


@app.route("/")
def index():
    page = render_template("index.html")
    return page

@app.route("/search", methods=["POST"])
def search_similar_weather_date():
    pass


@app.route('/get_options', methods=['POST'])
def get_options():
    data = request.get_json()
    date = data.get('date') # YYYY-MM-DD
    time = data.get('time') # hhmm
    print(date)
    print(time)
    date_time = pd.to_datetime(f"{date}T{time}", format="%Y-%m-%dT%H%M")
    print(date_time)
    nearest_topN_dates = INDEX.predict(DS.sel(time=[date_time]), n_top=5
    )
    nearest_topN_dates = [t.strftime("%Y%m%dT%H00") for t in nearest_topN_dates]
    print(nearest_topN_dates)
    return jsonify(nearest_topN_dates)



if __name__=="__main__":
    import os
    print(os.getcwd())
    app.run()
