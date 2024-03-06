from fastapi import FastAPI
import pandas as pd
from darts import TimeSeries
import pickle
from pydantic import BaseModel
from typing import Dict, Union, List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


class PredictionRequest(BaseModel):
    sku_id_str: str
    prediction_frequency: str


class PredictionResponse(BaseModel):
    pred_data: List[Dict[str, Union[str, float]]]
    last_30_data: List[Dict[str, Union[str, float]]]
    
# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the list of allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def predict_sales(request: PredictionRequest):

    sku_id_str = request.sku_id_str
    fr = request.prediction_frequency
    print("fr:", fr)
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

    # Load the model
    if fr == 'D':
        model_filename = "xgb_dynamic_pricing_model.pkl"
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        dataframe = pd.read_csv('dataset.csv')
        train = pd.read_csv('train.csv')
        pred_chunk = 30

    elif fr == 'W':
        model_filename = "catboost_dynamic_pricing_model.pkl"
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        dataframe = pd.read_csv('dataset_weekly.csv')
        train = pd.read_csv('train_weekly.csv')
        pred_chunk = 4

    future_features = ['base_price', 'is_holiday', 'day_of_week', 'week_of_month', 'month_of_year', 'days_till_black_friday', 'days_till_christmas', 'days_till_summer',
                       'days_till_winter', 'is_promotion', 'days_till_thanksgiving', 'days_till_independence_day', 'base_price_rolling_3', 'base_price_rolling_7', 'base_price_rolling_30']
    past_features = ['base_price', 'is_holiday', 'day_of_week', 'week_of_month', 'month_of_year', 'days_till_black_friday', 'days_till_christmas', 'days_till_summer',
                     'days_till_winter', 'is_promotion', 'days_till_thanksgiving', 'days_till_independence_day', 'base_price_rolling_3', 'base_price_rolling_7', 'base_price_rolling_30']
    train_time_series = TimeSeries.from_group_dataframe(
        train, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=['sales'])
    future_covariates_series2 = TimeSeries.from_group_dataframe(
        dataframe, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=future_features)
    past_covariates_series2 = TimeSeries.from_group_dataframe(
        dataframe, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=past_features)

    pred = loaded_model.predict(pred_chunk, series=train_time_series[sku_id], future_covariates=future_covariates_series2[sku_id],
                                past_covariates=past_covariates_series2[sku_id])

    df1 = pred.pd_dataframe()
    df1['sales'] = df1['sales'].apply(
        lambda x: max(0, round(x))).astype(int).tolist()

    pred_data = [{'date': date.strftime('%Y-%m-%d'), 'sales': sales}
                 for date, sales in zip(df1.index, df1['sales'])]

    df2 = train_time_series[sku_id].pd_dataframe()
    df2_last_30_dates = df2.tail(30).index.tolist()
    df2_last_30_sales = df2.tail(30)['sales'].tolist()

    last_30_data = [{'date': date.strftime('%Y-%m-%d'), 'sales': sales}
                    for date, sales in zip(df2_last_30_dates, df2_last_30_sales)]

    print(last_30_data)
    return PredictionResponse(pred_data=pred_data, last_30_data=last_30_data)
