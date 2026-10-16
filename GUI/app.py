from flask import Flask, render_template, request
import pandas as pd
import joblib
Data_path="census.csv"#this is the full data not the reduced version
adamodel_path="reduced_adaboost_model_mina.pkl"
gbmodel_path="reduced_GBmodel_heba.pkl"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
