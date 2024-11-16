import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataValidationArtifact,
)
from LoanPrediction.entity.config_entity import ModelEvaluationConfig
from LoanPrediction.utils.common import write_yaml_file, save_object, load_object
from sklearn.preprocessing import LabelEncoder
from LoanPrediction.utils.metric import get_classification_metric
from LoanPrediction.constant.constants import RESULT
from LoanPrediction.utils.model.estimator import LoanModel, ModelResolver
from LoanPrediction.pipeline.preprocessing import preprocessing_pipeline


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            logging.info("ModelEvaluation object initialized.")
        except Exception as e:
            logging.error(f"Error initializing ModelEvaluation: {e}")
            raise LoanException(e, sys.exc_info())

    def handle_outlier(
        self,
        df: pd.DataFrame,
        continuous_columns: list,
        remove_outliers=True,
        cap_outliers=False,
        z_threshold=3,
        iqr_factor=1.5,
    ) -> pd.DataFrame:
        try:
            logging.info("Handling outliers in continuous columns.")
            outliers_summary = []

            # Z-score method for outlier detection
            for col in continuous_columns:
                logging.info(f"Detecting outliers for column '{col}' using Z-score.")
                z_scores = stats.zscore(df[col])
                outliers_col_z = df[abs(z_scores) > z_threshold]
                outliers_summary.append(
                    {"column": col, "z_outliers": outliers_col_z.shape[0]}
                )

            # IQR Method for outlier detection
            for col in continuous_columns:
                logging.info(f"Detecting outliers for column '{col}' using IQR.")
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + iqr_factor * IQR
                lower_bound = Q1 - iqr_factor * IQR
                if remove_outliers:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    logging.info(
                        f"Outliers removed from column '{col}' using IQR thresholds."
                    )
                elif cap_outliers:
                    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                    logging.info(
                        f"Outliers capped in column '{col}' using IQR thresholds."
                    )

            logging.info("Outlier handling completed.")
            return df

        except Exception as e:
            logging.error(f"Error handling outliers: {e}")
            raise LoanException(f"Error handling outliers: {e}", sys.exc_info())

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading file '{file_path}': {e}")
            raise LoanException(e, sys.exc_info())

    @staticmethod
    def label_encode_target(series: pd.Series) -> pd.Series:
        try:
            if series.empty:
                raise ValueError("The input series is empty.")

            logging.info("Label encoding the target variable.")
            le = LabelEncoder()
            encoded_series = le.fit_transform(series)

            logging.info(f"Classes in the target variable: {list(le.classes_)}")
            return pd.Series(encoded_series, name=series.name)
        except Exception as e:
            logging.error(f"Error during label encoding: {e}")
            raise LoanException(e, sys.exc_info())

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation.")
            os.makedirs(
                self.model_evaluation_config.model_evaluation_dir, exist_ok=True
            )
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = ModelEvaluation.read_data(valid_train_file_path)
            test_df = ModelEvaluation.read_data(valid_test_file_path)

            df = pd.concat([train_df, test_df])
            if df.empty:
                raise LoanException("Concatenated DataFrame is empty.", sys.exc_info())
            logging.info("Data loaded and concatenated successfully.")

            continuous_columns = df.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()
            logging.info(f"Continuous columns identified: {continuous_columns}")

            df = self.handle_outlier(
                df, continuous_columns, remove_outliers=True, cap_outliers=False
            )

            X_data = df.drop(columns=[RESULT], axis=1)
            X_data = preprocessing_pipeline.transform(X_data)
            y_true = df[RESULT]
            y_true = ModelEvaluation.label_encode_target(y_true)

            model_train_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()

            is_model_accepted = True
            if not model_resolver.is_model_exists():
                logging.info("No existing model found. Using the trained model.")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_path=model_train_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None,
                )
                logging.info("Model evaluation completed.")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=model_train_file_path)

            logging.info("Comparing trained model with the latest model.")
            y_trained_pred = train_model.predict(X_data)
            y_latest_pred = latest_model.predict(X_data)

            trained_metric = get_classification_metric(y_true, y_trained_pred)
            latest_metric = get_classification_metric(y_true, y_latest_pred)

            improved_accuracy = trained_metric.f1_score - latest_metric.f1_score
            if self.model_evaluation_config.change_threshold < improved_accuracy:
                is_model_accepted = True
                logging.info(
                    f"Model accepted with improved accuracy: {improved_accuracy}"
                )
            else:
                is_model_accepted = False
                logging.info(f"Model rejected. Improved accuracy: {improved_accuracy}")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=model_train_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric,
            )

            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(
                self.model_evaluation_config.report_file_path, model_eval_report
            )
            logging.info("Model evaluation report saved.")
            return model_evaluation_artifact

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise LoanException(e, sys.exc_info())
