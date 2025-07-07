from sensor.exception import SensorException
from sensor.logger import logging
import os
import sys
from pandas import DataFrame
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.data_access.sensor_data import SensorData
from sklearn.model_selection import train_test_split

from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB to feature store")
            # Debug: confirm which collection we’re fetching
            print(f"🔍 Fetching from collection: {self.data_ingestion_config.collection_name}")

            sensor_data = SensorData()
            dataframe = sensor_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            # Validate non-empty
            if dataframe.shape[0] == 0:
                raise ValueError("No data fetched from MongoDB. DataFrame is empty.")

            # Write to feature store CSV
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise SensorException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        try:
            if dataframe.shape[0] < 2:
                raise ValueError("Not enough data to perform train-test split.")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train-test split on the dataframe")

            # Save train and test CSVs
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file paths")

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()

            # Drop configured columns safely
            drop_cols = self._schema_config.get("drop_columns", [])
            dataframe = dataframe.drop(drop_cols, axis=1, errors='ignore')

            self.split_data_as_train_test(dataframe=dataframe)

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
        except Exception as e:
            raise SensorException(e, sys)
