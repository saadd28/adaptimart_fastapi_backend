# mongodb://localhost:27017
import pandas as pd
import pymongo
from configparser import ConfigParser
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import holidays
import logging


def get_mongodb_connection(host, port, database):
    try:
        # Create MongoDB connection string
        connection_string = f"mongodb://{host}:{port}/{database}"

        # Connect to MongoDB
        client = pymongo.MongoClient(connection_string)
        db = client[database]

        return db
    except Exception as e:
        logging.error(f"Error connecting to MongoDB database: {e}")
        return None


def fetch_data(collection_name, db):
    try:
        cursor = db[collection_name].find()
        data = pd.DataFrame(list(cursor))
        return data
    except Exception as e:
        logging.error(
            f"Error fetching data from collection '{collection_name}': {e}")
        return None
    # try:
    #     cursor = db[collection_name].find({"SiteId": site_id})
    #     data = pd.DataFrame(list(cursor))
    #     return data
    # except Exception as e:
    #     logging.error(
    #         f"Error fetching data from collection '{collection_name}': {e}")
    #     return None


def fetch_configurations(collection_name, db, default_configurations):
    try:
        # Check if the configurations collection exists
        if collection_name not in db.list_collection_names():
            # If it doesn't exist, create it with default configurations
            if default_configurations is None:
                logging.error("Default configurations not provided.")
                return None
            logging.info("Creating Config Vars Collection in MongoDB")
            db.create_collection(collection_name)
            default_configurations = {
                "site_id": "201cb789-4198-488b-a5eb-4e7df0fb4bee",
                "threshold_base_price_change": 5,
                "threshold_minimum_sales": 200,
                "threshold_recent_months": 20,
                "frequency": 1
            }
            db[collection_name].insert_one(default_configurations)
            return default_configurations
        else:
            # Configurations collection exists, fetch the configurations
            logging.info("Config Vars Collection Already Exists")
            cursor = db[collection_name].find().limit(1)
            return cursor[0]
    except Exception as e:
        logging.error(
            f"Error fetching data from collection '{collection_name}': {e}")
        return None


def fetch_tenants_list(collection_name, db):
    try:
        # Check if the tenants collection exists
        if collection_name not in db.list_collection_names():
            logging.error(f"Collection '{collection_name}' does not exist.")
            return None
        else:
            # Tenants collection exists, fetch the tenants_list
            cursor = db[collection_name].find()
            tenants_list = [doc for doc in cursor]
            return tenants_list
    except Exception as e:
        logging.error(
            f"Error fetching data from collection '{collection_name}': {e}")
        return None


def extract_site_ids(tenants_list):
    site_ids = [tenant['site_id'] for tenant in tenants_list]
    return site_ids


def data_amputation(df, frequency):
    """_summary_

    Args:
        df (_type_): _description_
        frequency (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        # Group by product_item_sku_id and resampling day-wise, then forward fill
        feature_frequency_base_price_df = df.groupby('product_item_sku_id')['base_price'].resample(
            frequency).mean().groupby('product_item_sku_id').ffill().reset_index()
        feature_frequency_sale_price_df = df.groupby('product_item_sku_id')['sale_price'].resample(
            frequency).mean().groupby('product_item_sku_id').ffill().reset_index()
        feature_frequency_quantity_df = df.groupby('product_item_sku_id')['sales'].resample(
            frequency).sum().groupby('product_item_sku_id').fillna(0).reset_index()
        feature_frequency_list_price_df = df.groupby('product_item_sku_id')['list_price'].resample(
            frequency).mean().groupby('product_item_sku_id').ffill().reset_index()
        feature_frequency_msrp_price_df = df.groupby('product_item_sku_id')['msrp'].resample(
            frequency).mean().groupby('product_item_sku_id').ffill().reset_index()
        feature_frequency_views_df = df.groupby('product_item_sku_id')['views'].resample(
            frequency).sum().groupby('product_item_sku_id').fillna(0).reset_index()
        feature_frequency_carts_df = df.groupby('product_item_sku_id')['cart_quantity'].resample(
            frequency).sum().groupby('product_item_sku_id').fillna(0).reset_index()

        # Compiling Resampled Data
        data = pd.DataFrame(
            {
                'creation_date': feature_frequency_base_price_df['creation_date'],
                # 'site_id': feature_frequency_site_id_df['site_id'],
                'product_item_sku_id': feature_frequency_base_price_df['product_item_sku_id'],
                'sales': feature_frequency_quantity_df['sales'],
                'base_price': feature_frequency_base_price_df['base_price'],
                'list_price': feature_frequency_list_price_df['list_price'],
                'sale_price': feature_frequency_sale_price_df['sale_price'],
                'msrp': feature_frequency_msrp_price_df['msrp'],
                'views': feature_frequency_views_df['views'],
                'cart_quantity': feature_frequency_carts_df['cart_quantity']
            })

        return data
    except Exception as e:
        logging.error(f"Error in data amputation function: {e}")
        return None


def preprocessing_data(df, site_id, frequency_list, frequency):
    try:
        # print(df.columns)
        # Lowering Case Column Names
        df.columns = [col.lower() for col in df.columns]

        # Mapping Column Names
        column_mappings = {
            'creation_date': 'creation_date',
            'product_item_sku_id': 'product_item_sku_id',
            'quantity': 'sales',
            'base_price': 'base_price',
            'listprice': 'list_price',
            'saleprice': 'sale_price',
            'msrp': 'msrp',
            'views': 'views',
            'cart_quantity': 'cart_quantity',
            'siteid': 'site_id'
        }
        df = df.rename(columns=column_mappings)
        # print(df.head(10))

        df = df[df['site_id'] == site_id].copy()

        # Filtering Useful Columns
        df = df[[
            'creation_date',
            'product_item_sku_id',
            'base_price',
            'sales',
            'list_price',
            'sale_price',
            'msrp',
            'views',
            'cart_quantity',
            'site_id'
        ]]

        # Formatting Date
        df['creation_date'] = pd.to_datetime(df['creation_date'])
        df['creation_date'] = df['creation_date'].dt.date
        df.set_index('creation_date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        df.index = pd.to_datetime(df.index)

        print("Before Amputation:")
        df.info()
        data = data_amputation(df, frequency)

        print("After Amputation:")
        data['site_id'] = df['site_id'].iloc[0]
        # data.to_csv('data.csv', index=False, mode='w')

        data.info()
        return data
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        return None


def filter_on_base_price_change(df, threshold_base_price_change):
    try:
        # Identify changes in the base price for each product_item_sku_id
        df['base_price_change'] = df.groupby('product_item_sku_id')[
            'base_price'].diff().ne(0)

        # Count how many times the base price changed for each product_item_sku_id
        change_counts = df.groupby('product_item_sku_id')[
            'base_price_change'].sum().reset_index(name='no_of_price_changes')

        # Identify product_item_sku_ids where the base price changed at least 3 times
        sku_ids_to_filter = change_counts[change_counts['no_of_price_changes'] < (
            threshold_base_price_change + 1)]['product_item_sku_id']

        # Filter out rows for the identified product_item_sku_ids
        filtered_df = df[~df['product_item_sku_id'].isin(sku_ids_to_filter)]

        return filtered_df
    except Exception as e:
        logging.error(f"Error in Filtering on Base Price Change: {e}")
        return None


def filter_on_minimum_sales(df, threshold_minimum_sales):
    try:
        total_sales_per_sku = df.groupby('product_item_sku_id')[
            'sales'].sum().reset_index()
        skus_above_threshold = total_sales_per_sku[total_sales_per_sku['sales']
                                                   >= threshold_minimum_sales]

        filtered_df_sales = df[df['product_item_sku_id'].isin(
            skus_above_threshold['product_item_sku_id'])]

        return filtered_df_sales
    except Exception as e:
        logging.error(f"Error in Filtering on Minimum Sales: {e}")
        return None


def filter_on_minimum_days_sold(df, threshold_minimum_days_sold):
    try:
        total_sales_per_sku = df.groupby('product_item_sku_id')[
            'sales'].sum().reset_index()
        skus_above_threshold = total_sales_per_sku[total_sales_per_sku['sales']
                                                   >= threshold_minimum_days_sold]

        filtered_df_sales = df[df['product_item_sku_id'].isin(
            skus_above_threshold['product_item_sku_id'])]

        return filtered_df_sales
    except Exception as e:
        logging.error(f"Error in Filtering on Minimum Days Sold: {e}")
        return None


def filter_on_recent_months(df, threshold_recent_months):
    try:
        # Convert 'creation_date' to datetime format
        df['creation_date'] = pd.to_datetime(
            df['creation_date'], format='%Y-%m-%d')

        # Calculate the threshold date
        threshold_date = datetime.now() - timedelta(days=threshold_recent_months * 30)

        # Filter out SKUs with the latest sale being more than threshold months from system date
        filtered_skus = df.groupby('product_item_sku_id')[
            'creation_date'].max() > threshold_date
        filtered_skus = filtered_skus[filtered_skus].index.tolist()

        filtered_dataframe = df[df['product_item_sku_id'].isin(filtered_skus)]

        return filtered_dataframe
    except Exception as e:
        logging.error(f"Error in Filtering on Recent Months: {e}")
        return None


def threshold_filtering_price_optimization(df, threshold_base_price_change, threshold_minimum_sales, threshold_recent_months):
    try:
        filtered_df = filter_on_base_price_change(
            df, threshold_base_price_change)

        filtered_df_sales = filter_on_minimum_sales(
            filtered_df, threshold_minimum_sales)

        filtered_df_months = filter_on_recent_months(
            filtered_df_sales, threshold_recent_months)

        total_sales = filtered_df_months.groupby(
            'product_item_sku_id')['sales'].sum()
        # Add the total sales as a new column to the dataframe
        filtered_df_months['total_sales'] = filtered_df_months['product_item_sku_id'].map(
            total_sales)

        label_encoder = LabelEncoder()

        filtered_df_months['product_item_sku_id_encoded'] = label_encoder.fit_transform(
            filtered_df_months['product_item_sku_id'])

        cutoff_date = '2021-04-01'  # Replace with your desired date
        filtered_df_months = filtered_df_months[filtered_df_months['creation_date'] >= cutoff_date]
        # filtered_df_months.to_csv('thresholding_final.csv', index=False, mode='w')

        return filtered_df_months
    except Exception as e:
        logging.error(f"Error in Threshold Filtering: {e}")
        return None


def feature_engineering(df):
    try:

        df = df.reset_index(drop=True)
        df.info()
        # Add a column for holidays
        us_holidays = holidays.UnitedStates()
        df['is_holiday'] = [
            date in us_holidays for date in df['creation_date']]
        # Assuming 'df' is your DataFrame
        df['is_holiday'] = df['is_holiday'].map({False: 0, True: 1})
        df['creation_date'] = pd.to_datetime(df['creation_date'])
        df['day_of_week'] = df['creation_date'].dt.dayofweek
        df['week_of_month'] = df['creation_date'].apply(
            lambda x: (x.day-1)//7 + 1)
        df['month_of_year'] = df['creation_date'].dt.month
        # Add days till specific events

        def days_till_event(event_date, creation_date):
            event_month_day = event_date[1:]
            event_date = pd.to_datetime(
                f"{creation_date.year}-{event_month_day}")
            if creation_date > event_date:
                event_date = pd.to_datetime(
                    f"{creation_date.year + 1}-{event_month_day}")
            return (event_date - creation_date).days

        # add days till labor day, 4th of july, thanksgiving, newyears
        df['days_till_black_friday'] = df['creation_date'].apply(
            lambda x: days_till_event('-11-24', x))
        df['days_till_christmas'] = df['creation_date'].apply(
            lambda x: days_till_event('-12-25', x))
        df['days_till_summer'] = df['creation_date'].apply(
            lambda x: days_till_event('-06-01', x))
        df['days_till_winter'] = df['creation_date'].apply(
            lambda x: days_till_event('-12-01', x))
        df['days_till_independence_day'] = df['creation_date'].apply(
            lambda x: days_till_event('-06-04', x))
        df['days_till_thanksgiving'] = df['creation_date'].apply(
            lambda x: days_till_event('-11-28', x))
        df['is_promotion'] = df['sale_price'].notna().astype(int)
        # Calculate profit margin assuming 100% when base_price is half of msrp
        df['margin'] = ((df['msrp'] - df['base_price']) /
                        (df['msrp'] / 2)) * 100
        # Print the updated DataFrame
        df['creation_date'] = df['creation_date'].dt.strftime('%Y-%m-%d')
        # df.to_csv('feature_engineering.csv', index=False, mode='w')
        # Implement a rolling window (e.g., window size of 3 days)
        rolling_window_size = 3
        df['sales_rolling_3'] = df.groupby('product_item_sku_id')['sales'].rolling(
            window=rolling_window_size).mean().reset_index(drop=True)
        df['base_price_rolling_3'] = df.groupby('product_item_sku_id')['base_price'].rolling(
            window=rolling_window_size).mean().reset_index(drop=True)
        # Implement a rolling window (e.g., window size of 7 days)
        rolling_window_size = 7
        df['sales_rolling_7'] = df.groupby('product_item_sku_id')['sales'].rolling(
            window=rolling_window_size).mean().reset_index(drop=True)
        df['base_price_rolling_7'] = df.groupby('product_item_sku_id')['base_price'].rolling(
            window=rolling_window_size).mean().reset_index(drop=True)
        # Implement a rolling window (e.g., window size of 30 days)
        rolling_window_size = 30
        df['sales_rolling_30'] = df.groupby('product_item_sku_id')['sales'].rolling(
            window=rolling_window_size).mean().reset_index(drop=True)
        df['base_price_rolling_30'] = df.groupby('product_item_sku_id')['base_price'].rolling(
            window=rolling_window_size).mean().reset_index(drop=True)
        # df.to_csv('feature_engineering.csv', index=False, mode='w')
        return df
    except Exception as e:
        logging.error(f"Error in Feature Engineering: {e}")
        return None


def store_dataframe_in_mongodb(df, collection_name, db):
    try:
        # Convert DataFrame to list of dictionaries
        data = df.to_dict(orient='records')

        # Specify the collection
        collection = db[collection_name]

        # Insert documents into the collection
        collection.insert_many(data)

    except Exception as e:
        logging.error(f"An error storing feature engineered data: {e}")


def feature_engineering_collection_management(collection_name, db):
    try:
        # Specify the collection
        collection = db[collection_name]

        # Check if the collection exists
        if collection.count_documents({}) > 0:
            # Delete all documents from the collection
            collection.delete_many({})
        else:
            # If the collection doesn't exist, create one
            db.create_collection(collection_name)

    except Exception as e:
        logging.error(
            f"An error occurred while storing feature engineered data: {e}")
