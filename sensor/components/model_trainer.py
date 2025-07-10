from sensor.utils.main_utils import load_numpy_array_data, save_object, load_object
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel

from xgboost import XGBClassifier
import os
import sys



class ModelTrainer:
    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Initialize ModelTrainer with configuration and artifacts.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)




    def perform_hyper_paramter_tunig(self):
        """
        Hyperparameter tuning placeholder. Can be implemented using GridSearchCV or RandomizedSearchCV.
        """
        raise NotImplementedError("Hyperparameter tuning not yet implemented.")



    def train_model(self, x_train, y_train):
        """
        Train the XGBoost model on training data.
        """
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates model training, evaluation, and saving the final model.
        """
        try:

            # Load transformed arrays
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )


            # Train the model
            model = self.train_model(x_train, y_train)


            # Evaluate on training data
            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model does not meet the expected accuracy threshold.")


            # Evaluate on testing data
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)


            # Overfitting/Underfitting check
            score_diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if score_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model shows signs of overfitting/underfitting.")


            # Load preprocessor and save model
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=sensor_model)



            # Create and return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )


            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)

