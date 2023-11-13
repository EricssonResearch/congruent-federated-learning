from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

EPSILON = 1e-18


class ClassificationEvaluator(object):
    def __init__(self, tasks):
        self.tasks = tasks

    def evaluate(self, y, y_hat):
        error_dict_task = {}
        for task in self.tasks:
            if len(np.unique(y)) == 2:
                error_dict = {'acc': accuracy_score(y, np.argmax(y_hat, 1)),
                              'auc': roc_auc_score(y, np.argmax(y_hat, 1)),
                              'f1-score':  f1_score(y, np.argmax(y_hat, 1))}
            else:
                error_dict = {'acc': accuracy_score(y, np.argmax(y_hat, 1)),
                              'f1-score':  f1_score(y, np.argmax(y_hat, 1), average='macro')}
            error_dict_task.update({task: error_dict})
        return error_dict_task
