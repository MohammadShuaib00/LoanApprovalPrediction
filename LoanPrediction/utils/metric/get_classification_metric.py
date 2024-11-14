import os, sys
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from LoanPrediction.entity.artifact_entity import ClassificationMetricArtifact


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:

        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
        )
        return classification_metric
    except Exception as e:
        raise LoanException(e, sys.exc_info())
