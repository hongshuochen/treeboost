"""
Demo for defining a custom regression objective and metric
==========================================================

Demo for defining customized metric and objective.  Notice that for simplicity reason
weight is not used in following example. In this script, we implement the Logistic Regression
objective and logloss metric as customized functions, then compare it with
native implementation in XGBoost.

See :doc:`/tutorials/custom_metric_obj` for a step by step walkthrough, with other
details.

The `SLE` objective reduces impact of outliers in training dataset, hence here we also
compare its performance with standard squared error.

"""
import argparse
from time import time
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer

import xgboost as xgb

kRatio = 0.7
kSeed = 1994

kBoostRound = 100

np.random.seed(seed=kSeed)

def native_logistic(dtrain: xgb.DMatrix,
                 dtest: xgb.DMatrix) -> Dict[str, Dict[str, List[float]]]:
    '''Train using native implementation of Logistic Regression.'''
    print('Logistic Regression')
    results: Dict[str, Dict[str, List[float]]] = {}
    logistic = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'seed': kSeed,
        'max_depth': 6
    }
    start = time()
    xgb.train(logistic,
              dtrain=dtrain,
              num_boost_round=kBoostRound,
              evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
              evals_result=results)
    print('Finished Logistic Regression in:', time() - start)
    return results


def py_logistic(dtrain: xgb.DMatrix, dtest: xgb.DMatrix) -> Dict:
    '''Train using Python implementation of Logistic Regression.'''
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient for logistic regression.'''
        y = dtrain.get_label()
        return predt - y

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for logistic regression.'''
        return predt * (1.0 - predt)

    def logistic(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Logistic Regression objective. A simplified version for logistic used as
        objective function.

        # :math:`\frac{1}{2}[log(pred + 1) - log(label + 1)]^2`

        '''
        predt = 1.0 / (1.0 + np.exp(-predt))
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess

    def logloss(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        y = dtrain.get_label()
        predt = 1.0 / (1.0 + np.exp(-predt))
        return 'Pylogloss', -float(np.mean(y*np.log(predt) + (1-y)*np.log(1-predt)))

    def evalerror(preds, dtrain):
        labels = dtrain.get_label()
        # return a pair metric_name, result. The metric name must not contain a
        # colon (:) or a space since preds are margin(before logistic
        # transformation, cutoff at 0)
        return 'my-error', float(sum(labels != (preds > 0.0))) / len(labels)

    results: Dict[str, Dict[str, List[float]]] = {}
    start = time()
    xgb.train({'tree_method': 'gpu_hist', 'seed': kSeed, 'max_depth': 6, 'base_score': 0,
               'disable_default_eval_metric': 1},
              dtrain=dtrain,
              num_boost_round=kBoostRound,
              obj=logistic,
              custom_metric=logloss,
              evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
              evals_result=results)
    print('Finished Logistic Regression in:', time() - start)
    return results

def main(args):
    X, y = load_breast_cancer(return_X_y=True)
    # shuffle and split training and test sets
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    train_size = int(X.shape[0] * kRatio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    logistic_evals = native_logistic(dtrain, dtest)
    py_logistic_evals = py_logistic(dtrain, dtest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments for custom logistic objective function demo.')
    args = parser.parse_args()
    main(args)