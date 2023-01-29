import os
from datetime import datetime

import dill
import pandas as pd
import json


from modules.pipeline import model_filename
from modules.pipeline import path


def predict():

    dt = datetime.now().strftime("%Y%m%d%H%M")

    filename_model = model_filename
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

    filename_data_preds = f'{path}/data/predictions/preds_{dt}.csv'
    df_preds.to_csv(filename_data_preds, index=False)


if __name__ == '__main__':
    predict()
