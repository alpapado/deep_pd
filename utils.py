import os
import math
import pickle
import numpy as np

from datetime import datetime

from iproglearn.analytics.users import get_user_doc

def get_controlled_serials():
    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))
    controlled_serials = []
    for serial in subject_metadata:
        if "med_eval_1" in subject_metadata[serial]:
            controlled_serials.append(serial)
    return controlled_serials


def get_self_reports(serials):
    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))
    user_self_reports = {}
    for serial in serials:
        user_doc = subject_metadata[serial]
        if user_doc['healthstatus_id'] == 0:
            user_self_reports[serial] = 1
        else:
            user_self_reports[serial] = 0

    return user_self_reports

def conf_to_metrics(conf):
    tn, fp, fn, tp = conf.ravel()

    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    fscore = 2 * sensitivity * precision / (sensitivity + precision)
    metrics = {
               "accuracy": accuracy,
               "precision": precision,
               "sensitivity": sensitivity,
               "specificity": specificity,
               "fscore": fscore,
               }
    return metrics

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
