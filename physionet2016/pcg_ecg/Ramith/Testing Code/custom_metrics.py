import matplotlib.pyplot as plt
import neptune
from neptunecontrib.monitoring.utils import send_figure
import numpy as np
import pandas as pd
import scikitplot.metrics as plt_metrics
from scikitplot.helpers import binary_ks_curve
import seaborn as sns
import sklearn.metrics as sk_metrics

def _class_metrics(y_true, y_pred_class):
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred_class).ravel()

    true_positive_rate = tp / (tp + fn)
    true_negative_rate = tn / (tn + fp)
    positive_predictive_value = tp / (tp + fp)
    negative_predictive_value = tn / (tn + fn)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (tp + fn)
    false_discovery_rate = fp / (tp + fp)

    scores = {'accuracy': sk_metrics.accuracy_score(y_true, y_pred_class),
              'precision': sk_metrics.precision_score(y_true, y_pred_class),
              'recall': sk_metrics.recall_score(y_true, y_pred_class),
              'f1_score': sk_metrics.fbeta_score(y_true, y_pred_class, beta=1),
              'f2_score': sk_metrics.fbeta_score(y_true, y_pred_class, beta=2),
              'matthews_corrcoef': sk_metrics.matthews_corrcoef(y_true, y_pred_class),
              'cohen_kappa': sk_metrics.cohen_kappa_score(y_true, y_pred_class),
              'true_positive_rate': true_positive_rate,
              'true_negative_rate': true_negative_rate,
              'positive_predictive_value': positive_predictive_value,
              'negative_predictive_value': negative_predictive_value,
              'false_positive_rate': false_positive_rate,
              'false_negative_rate': false_negative_rate,
              'false_discovery_rate': false_discovery_rate}

    return scores