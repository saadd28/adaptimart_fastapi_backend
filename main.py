# mongodb://localhost:27017
import pandas as pd
import logging
import utils
from configparser import ConfigParser
import ast


def run_main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    config_file = "config.ini"

    parser = ConfigParser()
    parser.read(config_file)

    # Read MongoDB connection parameters
    host = parser.get('mongodb', 'host')
    port = int(parser.get('mongodb', 'port'))
    database = parser.get('mongodb', 'database')
    frequency_list = parser.get('mongodb', 'frequency_list')
    frequency_list = ast.literal_eval(frequency_list)
    dynamic_pricing_data_collection_name = parser.get(
        'mongodb', 'dynamic_pricing_data_collection_name')
    config_collection = parser.get('mongodb', 'config_collection')
    tenants_collection_name = parser.get('mongodb', 'tenants_collection_name')
    feature_engineered_collection_name_daily = parser.get(
        'mongodb', 'feature_engineered_collection_name_daily')
    feature_engineered_collection_name_weekly = parser.get(
        'mongodb', 'feature_engineered_collection_name_weekly')
    feature_engineered_collection_name_monthly = parser.get(
        'mongodb', 'feature_engineered_collection_name_monthly')
    interval_days = int(parser.get(
        'mongodb', 'interval_days'))

    db = utils.get_mongodb_connection(host, port, database)

    dynamic_pricing_data = None
    configurations = None

    if db is not None:
        logging.info("MongoDB database connected successfully!")

        logging.info("Fetching Tenants List from MongoDB")
        tenants_list = utils.fetch_tenants_list(tenants_collection_name, db)
        site_ids = utils.extract_site_ids(tenants_list)

        if site_ids is not None:
            logging.info("Site Ids List: %s", site_ids)

        logging.info("Creating/Clearing Daily Feature Engineered Collection")
        utils.feature_engineering_collection_management(
            feature_engineered_collection_name_daily, db)

        logging.info("Creating/Clearing Weekly Feature Engineered Collection")
        utils.feature_engineering_collection_management(
            feature_engineered_collection_name_weekly, db)

        logging.info("Creating/Clearing Monthly Feature Engineered Collection")
        utils.feature_engineering_collection_management(
            feature_engineered_collection_name_monthly, db)

        for site_id in site_ids:
            logging.info("Fetching Configurations from MongoDB!")
            configurations = utils.fetch_configurations(
                config_collection, db, site_id)
            # Check if the collection exists
            if dynamic_pricing_data_collection_name not in db.list_collection_names():
                raise ValueError(
                    f"Collection '{configurations['collection_name']}' does not exist in the database.")

            for index, frequency in enumerate(frequency_list):
                logging.info(
                    "Data processing is starting for site ID: %s and frequency: %s", site_id, frequency)
                logging.info("Fetching Data from MongoDB...")
                dynamic_pricing_data = utils.fetch_data(
                    dynamic_pricing_data_collection_name, db)

                logging.info("Starting Preprocessing of data...")
                dynamic_pricing_data = utils.preprocessing_data(
                    dynamic_pricing_data, site_id, frequency)

                logging.info("Applying Threshold Filtering...")
                dynamic_pricing_data = utils.threshold_filtering_price_optimization(
                    dynamic_pricing_data, configurations["threshold_base_price_change"], configurations["threshold_minimum_sales"], configurations["threshold_recent_months"], interval_days, site_id)

                logging.info("Doing Feature Engineering...")
                dynamic_pricing_data = utils.feature_engineering(
                    dynamic_pricing_data)

                if (index == 0):
                    logging.info(
                        "Storing Daily Feature Engineered Data in MongoDB...")
                    utils.store_dataframe_in_mongodb(
                        dynamic_pricing_data, feature_engineered_collection_name_daily, db)
                    logging.info(
                        "Successfully Stored Daily Feature Engineered Data in MongoDB!")
                elif (index == 1):
                    logging.info(
                        "Storing Weekly Feature Engineered Data in MongoDB...")
                    utils.store_dataframe_in_mongodb(
                        dynamic_pricing_data, feature_engineered_collection_name_weekly, db)
                    logging.info(
                        "Successfully Stored Weekly Feature Engineered Data in MongoDB!")
                elif (index == 2):
                    logging.info(
                        "Storing Monthly Feature Engineered Data in MongoDB...")
                    utils.store_dataframe_in_mongodb(
                        dynamic_pricing_data, feature_engineered_collection_name_monthly, db)
                    logging.info(
                        "Successfully Stored Monthly Feature Engineered Data in MongoDB!")

    else:
        logging.error("Failed to connect to MongoDB database.")


if __name__ == "__main__":
    run_main()
