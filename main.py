from fastapi import FastAPI, Depends
import pandas as pd
from darts import TimeSeries
import pickle
from pydantic import BaseModel
from typing import Dict, Union, List
from fastapi.middleware.cors import CORSMiddleware
import logging
import pymongo
from functions import train_test_split_last_n_rows

app = FastAPI()
host = "localhost"
port = 27017
database = "dynamic_pricing_db"
collection_name_one = "feature_engineered_data"
collection_name_two = "feature_engineered_data_two"

# Create MongoDB connection string
connection_string = f"mongodb://{host}:{port}/{database}"
# Connect to MongoDB
client = pymongo.MongoClient(connection_string)
db = client[database]


# Dependency to inject MongoDB database client


def get_database():
    try:
        cursor = db[collection_name_one].find()
        daily_data = pd.DataFrame(list(cursor))

    except Exception as e:
        logging.error(
            f"Error fetching data from collection '{collection_name_one}': {e}")

    try:
        cursor = db[collection_name_two].find()
        weekly_data = pd.DataFrame(list(cursor))

    except Exception as e:
        logging.error(
            f"Error fetching data from collection '{collection_name_two}': {e}")

    return db, daily_data, weekly_data


class PredictionRequest(BaseModel):
    sku_id_str: str
    prediction_frequency: str
    site_id: str


class PredictionResponse(BaseModel):
    data: List[Dict[str, Union[str, float]]]


# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the list of allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def predict_sales(request: PredictionRequest, db: pymongo.database.Database = Depends(get_database),
                        daily_data: pd.DataFrame = Depends(
                            lambda: get_database()[1]),
                        weekly_data: pd.DataFrame = Depends(lambda: get_database()[2])):

    sku_id_str = request.sku_id_str
    fr = request.prediction_frequency
    site_id = request.site_id
    sku_mapping = {
        'FO-DMUN2Q': 0,
        'MA-DMDCNPLQ': 1,
        'MA-DMSUFQ': 2,
        'MA-DMSUFT': 3,
        'PI-DMDCFMJ': 4,
        'PI-DMDCFMK': 5,
        'PI-DMDCMEJ': 6,
        'PI-DMDCMEK': 7,
        'PI-DMDCSOJ': 8,
        'PI-DMDCSOK': 9
    }
    sku_id = sku_mapping.get(sku_id_str)
    data = pd.DataFrame()
    if fr == 'D':
        data = daily_data[daily_data['site_id'] == site_id]
        model_filename = "xgboost_daily_dynamic_pricing_model.pkl"
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        pred_chunk = 30

    elif fr == 'W':
        data = weekly_data[weekly_data['site_id'] == site_id]
        model_filename = "catboost_weekly_dynamic_pricing_model.pkl"
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        pred_chunk = 7

    train, test = train_test_split_last_n_rows(data, 0.1)

    future_features = ['base_price', 'is_holiday', 'day_of_week', 'week_of_month', 'month_of_year', 'days_till_black_friday', 'days_till_christmas', 'days_till_summer',
                       'days_till_winter', 'is_promotion', 'days_till_thanksgiving', 'days_till_independence_day', 'base_price_rolling_3', 'base_price_rolling_7', 'base_price_rolling_30']
    past_features = ['base_price', 'is_holiday', 'day_of_week', 'week_of_month', 'month_of_year', 'days_till_black_friday', 'days_till_christmas', 'days_till_summer',
                     'days_till_winter', 'is_promotion', 'days_till_thanksgiving', 'days_till_independence_day', 'base_price_rolling_3', 'base_price_rolling_7', 'base_price_rolling_30']

    train_time_series = TimeSeries.from_group_dataframe(
        train, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=['sales'])
    future_covariates_series2 = TimeSeries.from_group_dataframe(
        data, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=future_features)
    past_covariates_series2 = TimeSeries.from_group_dataframe(
        data, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=past_features)

    pred = loaded_model.predict(pred_chunk, series=train_time_series[sku_id], future_covariates=future_covariates_series2[sku_id],
                                past_covariates=past_covariates_series2[sku_id])

    df1 = pred.pd_dataframe()
    df1['sales'] = df1['sales'].apply(
        lambda x: max(0, round(x))).astype(int).tolist()

    df1['flag'] = 0
    df2 = train_time_series[sku_id].pd_dataframe().tail(30)
    df2['flag'] = 1
    result_df = pd.concat([df2, df1])
    data = [{'date': date.strftime('%Y-%m-%d'), 'sales': sales, 'flag': flag}
            for date, sales, flag in zip(result_df.index, result_df['sales'], result_df['flag'])]

    return PredictionResponse(data=data)
