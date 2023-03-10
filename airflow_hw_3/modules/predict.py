import os
from datetime import datetime

import dill
import pandas as pd
import json

from modules.pipeline import path
from modules.pipeline import path_models


def get_filename_of_last_model():
    filename_of_last_model = ''

    files = os.listdir(path_models)
    files = [file for file in files if os.path.isfile(os.path.join(path_models, file))]
    if files:
        files = [os.path.join(path_models, file) for file in files]
        filename_of_last_model = max(files, key=os.path.getctime)

    return filename_of_last_model


def predict():

    filename_model = get_filename_of_last_model()
    if filename_model == '':
        return
    with open(filename_model, 'rb') as file:
        model = dill.load(file)

    dir = f'{path}/data/test'

    df_preds = pd.DataFrame()
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), 'r') as file:
            dict = json.load(file)
        df = pd.DataFrame(dict, index=[0])
        df.insert(0, 'price_category', model.predict(df))
        df.insert(0, 'filename', filename)
        df_preds = df_preds.append(df)

    filename_data_preds = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_preds.to_csv(filename_data_preds, index=False)


if __name__ == '__main__':
    predict()
