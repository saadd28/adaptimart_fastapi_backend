# mongodb://localhost:27017
import pandas as pd
import logging
import utils
from configparser import ConfigParser


def run_main():
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    config_file = "config.ini"
    # collection_name = 'dynamic_pricing_data'
    # config_collection = 'config_vars'
    # feature_engineered_collection = 'feature_engineered_data'
    # Create a parser
    parser = ConfigParser()
    parser.read(config_file)

    # Read MongoDB connection parameters
    host = parser.get('mongodb', 'host')
    port = int(parser.get('mongodb', 'port'))
    database = parser.get('mongodb', 'database')
    frequency_list = parser.get('mongodb', 'frequency_list')
    frequency = parser.get('mongodb', 'frequency')
    dynamic_pricing_data_collection_name = parser.get(
        'mongodb', 'dynamic_pricing_data_collection_name')
    feature_engineered_collection_name = parser.get(
        'mongodb', 'feature_engineered_collection_name')
    config_collection = parser.get('mongodb', 'config_collection')
    tenants_collection_name = parser.get('mongodb', 'tenants_collection_name')

    default_configurations = {
        "site_id": "8d3ea3bc-f65b-4227-9fa6-6fae40e4575a",
        "threshold_base_price_change": 10,
        "threshold_minimum_sales": 480,
        "threshold_recent_months": 20,
        "frequency": 7
    }
    # site_id, end date is current date
    # interval days in ini and then start date calculate at run time
    # main in functions
    # tenant, site_id, product_id, sku_id

    db = utils.get_mongodb_connection(host, port, database)

    dynamic_pricing_data = None
    configurations = None

    if db is not None:
        logging.info("MongoDB database connected successfully!")

        logging.info("Fetching Tenants List from MongoDB")
        tenants_list = utils.fetch_tenants_list(tenants_collection_name, db)
        tenants_list = utils.extract_site_ids(tenants_list)

        if tenants_list is not None:
            logging.info("Tenants List: ", tenants_list)

        logging.info("Fetching Configurations from MongoDB!")
        configurations = utils.fetch_configurations(
            config_collection, db, default_configurations)
        # Check if the collection exists
        if dynamic_pricing_data_collection_name not in db.list_collection_names():
            raise ValueError(
                f"Collection '{configurations['collection_name']}' does not exist in the database.")

        utils.feature_engineering_collection_management(
            feature_engineered_collection_name, db)
        for site_id in tenants_list:
            logging.info("Fetching Data from MongoDB...")
            dynamic_pricing_data = utils.fetch_data(
                dynamic_pricing_data_collection_name, db)

            logging.info("Starting Preprocessing of data...")
            dynamic_pricing_data = utils.preprocessing_data(
                dynamic_pricing_data, site_id, frequency_list, frequency)

            logging.info("Applying Threshold Filtering...")
            dynamic_pricing_data = utils.threshold_filtering_price_optimization(
                dynamic_pricing_data, configurations["threshold_base_price_change"], configurations["threshold_minimum_sales"], configurations["threshold_recent_months"])

            logging.info("Doing Feature Engineering...")
            dynamic_pricing_data = utils.feature_engineering(
                dynamic_pricing_data)

            logging.info("Storing Feature Engineered Data in MongoDB...")
            utils.store_dataframe_in_mongodb(
                dynamic_pricing_data, feature_engineered_collection_name, db)

    else:
        logging.error("Failed to connect to MongoDB database.")
