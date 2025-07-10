import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object




class DataTransformation:

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Initialize DataTransformation with validation artifact and config.
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e, sys)



    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read CSV file into DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys)



    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Returns sklearn Pipeline object with imputer and scaler.
        """
        try:
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            robust_scaler = RobustScaler()

            preprocessor = Pipeline(steps=[
                ("Imputer", simple_imputer),     # Replace missing values with 0
                ("RobustScaler", robust_scaler)  # Scale to handle outliers
            ])

            return preprocessor

        except Exception as e:
            raise SensorException(e, sys) from e





    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute the data transformation pipeline:
        - Read data
        - Transform features
        - Apply SMOTETomek to training data
        - Save transformed arrays and preprocessing object
        - Return artifact
        """

        try:
            # Load datasets
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)


            # Separate input and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())


            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())


            # Preprocessing pipeline
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)


            # Transform input features
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)


            # Apply SMOTETomek on training data only
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )


            # Use test data as-is (no resampling)
            input_feature_test_final = transformed_input_test_feature
            target_feature_test_final = target_feature_test_df


            # Concatenate features and labels
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]


            # Save transformed data and preprocessing object
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, array=test_arr
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path, preprocessor_object
            )


            # Create and return artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
