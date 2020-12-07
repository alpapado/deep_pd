import glob
import pickle
from functools import partial

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from imu import (maybe_train_imu_models, prepare_imu_gdata,
                 prepare_imu_sdata)

from keyboard import (maybe_train_typing_models, prepare_typing_gdata,
                      prepare_typing_sdata)
from models import (AttentionMILTremorTime as TremorMILTime,
        AttentionMILRigidity as RigidityMIL,
        AttentionMILTremorFrequency as TremorMILFrequency,
        ModelFusionMultiLabel)
from utils import conf_to_metrics, freeze_model, get_self_reports, get_controlled_serials
from vat_pytorch.torch_utils import CustomDataset1M, CustomDataset2M
from vat_pytorch.vat import MultiLabelModel2M

print = partial(print, flush=True)


sacred_db_uri = 'mongodb_uri'
ex = Experiment('Deep MIL PD')

ex.observers.append(MongoObserver.create(sacred_db_uri))


@ex.config
def config():
    train_params = {
        "pre_training_epochs": 100,
        "finetuning_epochs": 1000,
        "batch_size": 8,
        "num_trials": 10,
        "use_weights": False,
        "decision_th": 0.5,
        "pooling": 'attention',
        "embedding_dim": 64,
        "mode": "multi_label",
        "evaluation_on": "sdata",
        "base_lr": 1e-3,
        "finetuning_lr": 5e-4,
        "fused_checkpoint": False,
    }

    imu_params = {
        "checkpoint": None,
        "bag_length": 1500,
        "min_windows": 5,
        "domain": 'time',
    }

    typing_params = {
        "checkpoint": None,
        "bag_length": 500,
        "target": "fmi",
        "min_sessions": 5,
        "ft_split_threshold_ms": 4000,
        "ht_split_threshold_ms": 1000,
        "min_keystrokes_per_session": 40,
        "use_development_set": True,
        "bin_width_ms": 10,
    }

    imu_sdataset = "imu_sdata.pickle"
    typing_sdataset = "typing_sdata.pickle"
    imu_gdataset = "imu_gdata.pickle"
    typing_gdataset = "typing_gdata.pickle"


def evaluate_ensemble(model_list, test_loader, decision_th):
    X_test_serials = test_loader.dataset.sid
    user_self_reports = get_self_reports(X_test_serials)
    per_subject_predictions = {}

    for serial in X_test_serials:
        per_subject_predictions[serial] = []

    for i, model in enumerate(model_list):
        print("Model ", i)
        probabilities, serials = model.predict(test_loader)

        if len(probabilities.shape) > 1:
            probabilities = probabilities[:, -1]
        y_score = probabilities

        for j, serial in enumerate(serials):
            per_subject_predictions[serial].append(y_score[j])

    # Evaluate using model ensemble
    print("#### Ensemble validation ####")
    y_true = []
    y_score = []

    for i, serial in enumerate(serials):
        y_score.append(np.mean(per_subject_predictions[serial]))
        y_true.append(user_self_reports[serial])

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Plot ROC curve
    auc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, 'b', label='AUC = %.2f' % auc_score)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.legend(loc='lower right')
    fig.savefig('artifacts/roc_curve.svg')

    result = {"y_true": y_true, "y_score": y_score, "serials": serials}
    pickle.dump(result, open("artifacts/gdata_result.pickle", "wb"))
    print("AUC = %.3f " % auc_score)

    return auc_score


def evaluate_loso_sdata(train_params, imu_params, typing_params,
                        imu_sdataset, typing_sdataset):
    mode = train_params["mode"]
    batch_size = train_params["batch_size"]
    embedding_dim = train_params["embedding_dim"]
    decision_th = train_params["decision_th"]
    finetuning_epochs = train_params["finetuning_epochs"]
    finetuning_lr = train_params["finetuning_lr"]

    X_imu, X_typing, X_masks_imu, X_masks_typing, y_imu, y_typing, y_pd, common_serials = get_common_sdata_subset(
        imu_sdataset, typing_sdataset, imu_params, typing_params)

    common_serials = np.array(common_serials)

    print("COMMON SERIALS ", len(common_serials))

    tremor_conf_fused = np.zeros((2, 2))
    fmi_conf_fused = np.zeros((2, 2))
    pd_conf_fused = np.zeros((2, 2))

    tremor_conf_standalone = np.zeros((2, 2))
    fmi_conf_standalone = np.zeros((2, 2))
    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))

    for j, left_out in enumerate(common_serials):
        user_doc = subject_metadata[left_out]
        med_eval_1 = user_doc["med_eval_1"]
        med_eval_2 = user_doc["med_eval_2"]
        updrs23_1 = med_eval_1["updrs_23_right"] + med_eval_1["updrs_23_left"]
        updrs23_2 = med_eval_2["updrs_23_right"] + med_eval_2["updrs_23_left"]
        updrs22_1 = med_eval_1["updrs_22_uextr_right"] + \
            med_eval_1["updrs_22_uextr_left"]
        updrs22_2 = med_eval_2["updrs_22_uextr_right"] + \
            med_eval_2["updrs_22_uextr_left"]
        updrs23 = updrs23_1 + updrs23_2
        updrs22 = updrs22_1 + updrs22_2
        fpp_fmi = med_eval_1["fpp_ams_fine_motor"] + med_eval_2["fpp_ams_fine_motor"]
        y_true_tremor = med_eval_1["tremor_manual_2_2_2020_new"]

        if typing_params["target"] == 'updrs22':
            y_true_fmi = int(updrs22 > 0)
        elif typing_params["target"] == 'updrs23':
            y_true_fmi = int(updrs23 > 0)
        elif typing_params["target"] == 'fmi':
            y_true_fmi = int(np.logical_or(np.logical_and(updrs22, updrs23), fpp_fmi))

        y_true_pd = med_eval_1["pd_status"]

        left_out_idx = np.where(common_serials==left_out)[0]

        print(j, " - Left-out ", left_out)
        print("y_tremor: ", y_true_tremor)
        print("y_fmiping: ", y_true_fmi)
        print("y_pd: ", y_true_pd)

        print("#### Pre-train models without left-out subject ####")
        tremor_model_list = maybe_train_imu_models(imu_sdataset, imu_params,
                                                   train_params,
                                                   serials_to_exclude=[left_out],
                                                   left_out=left_out)

        rigidity_model_list = maybe_train_typing_models(typing_sdataset,
                                                        typing_params,
                                                        train_params,
                                                        serials_to_exclude=[left_out],
                                                        left_out=left_out)
        train_idx = common_serials != left_out
        test_idx = common_serials == left_out

        X_imu_train = X_imu[train_idx]
        X_typing_train = X_typing[train_idx]
        X_masks_imu_train = X_masks_imu[train_idx]
        X_masks_typing_train = X_masks_typing[train_idx]
        y_train = np.column_stack(
            (y_imu[train_idx], y_typing[train_idx], y_pd[train_idx])).astype('float32')
        serials_train = common_serials[train_idx]

        X_imu_test = X_imu[test_idx]
        X_typing_test = X_typing[test_idx]
        X_masks_imu_test = X_masks_imu[test_idx]
        X_masks_typing_test = X_masks_typing[test_idx]
        y_test = np.column_stack(
            (y_imu[test_idx], y_typing[test_idx], y_pd[test_idx])).astype('float32')
        serials_test = common_serials[test_idx]

        print("#### Finetune models without left_out subject ####")
        train_set = CustomDataset2M(X_imu_train, X_typing_train, y_train,
                                    X_masks_imu_train, X_masks_typing_train,
                                    serials_train)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)

        test_set = CustomDataset2M(X_imu_test, X_typing_test, y_test,
                                   X_masks_imu_test, X_masks_typing_test,
                                   serials_test)
        test_loader = DataLoader(test_set, batch_size=200, shuffle=False)

        # Evaluate standalone models first
        test_set = CustomDataset1M(
            X_imu_test, y_test[:, 0], X_masks_imu_test, serials_test)
        test_loader_tremor = DataLoader(test_set, batch_size=200, shuffle=False)

        test_set = CustomDataset1M(
            X_typing_test, y_test[:, 1], X_masks_typing_test, serials_test)
        test_loader_fmi = DataLoader(test_set, batch_size=200, shuffle=False)

        print("Total subjects: ", y_train.shape[0])
        print("PD subjects: ", np.count_nonzero(y_train[:, -1] == 1))
        print("Healthy subjects: ", np.count_nonzero(y_train[:, -1] == 0))
        print()

        model_list = []

        this_tremor_conf_fused = np.zeros((2, 2))
        this_fmi_conf_fused = np.zeros((2, 2))
        this_pd_conf_fused = np.zeros((2, 2))

        for i in range(train_params["num_trials"]):
            tremor_model = tremor_model_list[i]
            rigidity_model = rigidity_model_list[i]

            probs_tr, _ = tremor_model.predict(test_loader_tremor)
            probs_fmi, _ = rigidity_model.predict(test_loader_fmi)
            pred_tr = (probs_tr >= decision_th).astype(int)
            pred_fmi = (probs_fmi >= decision_th).astype(int)

            tremor_conf_standalone += confusion_matrix(
                [y_true_tremor], pred_tr, labels=[0, 1])
            fmi_conf_standalone += confusion_matrix(
                [y_true_fmi], pred_fmi, labels=[0, 1])

            freeze_model(tremor_model)
            freeze_model(rigidity_model)

            combined_logit_model = ModelFusionMultiLabel(
                tremor_model.logit.g, rigidity_model.logit.g, embedding_dim)
            model = MultiLabelModel2M(
                model=combined_logit_model, lr=finetuning_lr)

            model.fit(train_loader,
                      test_loader=train_loader,
                      num_epochs=finetuning_epochs)
            model_list.append(model)

            probs, _ = model.predict(test_loader)
            predictions = (probs >= decision_th).astype(int)

            tremor_conf_fused += confusion_matrix(
                [y_true_tremor], predictions[:, 0], labels=[0, 1])
            fmi_conf_fused += confusion_matrix([y_true_fmi],
                                               predictions[:, 1], labels=[0, 1])
            pd_conf_fused += confusion_matrix([y_true_pd],
                                              predictions[:, -1], labels=[0, 1])

            this_tremor_conf_fused += confusion_matrix(
                [y_true_tremor], predictions[:, 0], labels=[0, 1])
            this_fmi_conf_fused += confusion_matrix(
                [y_true_fmi], predictions[:, 1], labels=[0, 1])
            this_pd_conf_fused += confusion_matrix(
                [y_true_pd], predictions[:, -1], labels=[0, 1])

        print(this_tremor_conf_fused)
        print(this_fmi_conf_fused)
        print(this_pd_conf_fused)
        print("")

    print("#### Pre-trained models ####")
    print("## Tremor ##")
    print(tremor_conf_standalone)
    print(conf_to_metrics(tremor_conf_standalone))

    print("## FMI ##")
    print(fmi_conf_standalone)
    print(conf_to_metrics(fmi_conf_standalone))

    print("#### After joint training ####")
    print("Tremor conf")
    print(tremor_conf_fused)

    print("FMI conf")
    print(fmi_conf_fused)

    print("PD conf")
    print(pd_conf_fused)

    tremor_metrics = conf_to_metrics(tremor_conf_fused)
    fmi_metrics = conf_to_metrics(fmi_conf_fused)
    pd_metrics = conf_to_metrics(pd_conf_fused)

    print("Tremor metrics")
    print(tremor_metrics)

    print("fmiping metrics")
    print(fmi_metrics)

    print("PD metrics")
    print(pd_metrics)

    return pd_metrics


def get_common_sdata_subset(imu_sdataset, typing_sdataset,
                            imu_params, typing_params):
    X_imu, X_masks_imu, y_imu, serials_imu = prepare_imu_sdata(
        imu_sdataset, imu_params)

    X_typing, X_masks_typing, y_typing, serials_typing = prepare_typing_sdata(
        typing_sdataset, typing_params)

    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))

    # Merge the modalities
    common_serials, idx_to_keep_imu, idx_to_keep_typing = np.intersect1d(
        serials_imu, serials_typing, return_indices=True)
    common_serials = list(common_serials)

    X_imu = X_imu[idx_to_keep_imu]
    X_masks_imu = X_masks_imu[idx_to_keep_imu]
    y_imu = y_imu[idx_to_keep_imu]
    serials_imu = serials_imu[idx_to_keep_imu]

    X_typing = X_typing[idx_to_keep_typing]
    X_masks_typing = X_masks_typing[idx_to_keep_typing]
    y_typing = y_typing[idx_to_keep_typing]
    serials_typing = serials_typing[idx_to_keep_typing]

    y_pd = []

    print("Training labels")
    for i, serial in enumerate(common_serials):
        user_doc = subject_metadata[serial]
        pd_status = user_doc["med_eval_1"]["pd_status"]
        print(serial, y_imu[i], y_typing[i], pd_status)
        y_pd.append(pd_status)

    y_pd = np.array(y_pd).astype('float32')

    return X_imu, X_typing, X_masks_imu, X_masks_typing, y_imu, y_typing, y_pd, common_serials


def get_common_gdata_subset(imu_gdataset, typing_gdataset,
                            imu_params, typing_params,
                            serials_to_exclude):
    X_imu, X_masks_imu, y_imu, serials_imu = prepare_imu_gdata(
        imu_gdataset, imu_params, serials_to_exclude)

    X_typing, X_masks_typing, y_typing, serials_typing = prepare_typing_gdata(
        typing_gdataset, typing_params, serials_to_exclude)

    # Merge the modalities
    common_serials, idx_to_keep_imu, idx_to_keep_typing = np.intersect1d(
        serials_imu, serials_typing, return_indices=True)

    common_serials = list(common_serials)
    y_pd = []

    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))
    for serial in common_serials:
        user_doc = subject_metadata[serial]
        y_pd.append(int(user_doc["healthstatus_id"] == 0))

    y_pd = np.array(y_pd).astype('float32')

    X_imu = X_imu[idx_to_keep_imu]
    X_masks_imu = X_masks_imu[idx_to_keep_imu]
    y_imu = y_imu[idx_to_keep_imu]
    serials_imu = serials_imu[idx_to_keep_imu]

    X_typing = X_typing[idx_to_keep_typing]
    X_masks_typing = X_masks_typing[idx_to_keep_typing]
    y_typing = y_typing[idx_to_keep_typing]
    serials_typing = serials_typing[idx_to_keep_typing]

    return X_imu, X_typing, X_masks_imu, X_masks_typing, y_pd, common_serials


def evaluate_pretrained_gdata(train_params, imu_params, typing_params,
                              imu_sdataset, typing_sdataset, imu_gdataset,
                              typing_gdataset):
    embedding_dim = train_params["embedding_dim"]
    batch_size = train_params["batch_size"]
    finetuning_epochs = train_params["finetuning_epochs"]
    decision_th = train_params["decision_th"]
    fused_checkpoint = train_params["fused_checkpoint"]

    print("#### Stage 1 - Pre-train symptom models independently ####")
    tremor_model_list = maybe_train_imu_models(imu_sdataset,
                                               imu_params,
                                               train_params)
    print("")

    rigidity_model_list = maybe_train_typing_models(typing_sdataset,
                                                    typing_params,
                                                    train_params)
    print("")

    print("#### Stage 2 - Freeze bag embeddings and finetune on common subjects ####")
    fused_models_path = "models/finetuned/gdata/"
    pretrained_models_path = "models/pretrained/gdata/"
    if fused_checkpoint is False:
        X_imu, X_typing, X_masks_imu, X_masks_typing, y_imu, y_typing, y_pd, serials = get_common_sdata_subset(
            imu_sdataset, typing_sdataset,
            imu_params, typing_params)

        y_pd = np.column_stack((y_imu, y_typing, y_pd)).astype('float32')

        train_set = CustomDataset2M(X_imu, X_typing, y_pd,
                                    X_masks_imu, X_masks_typing, serials)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)

        print("Total subjects: ", y_pd.shape[0])
        print("PD subjects: ", np.count_nonzero(y_pd[:, -1] == 1))
        print("Healthy subjects: ", np.count_nonzero(y_pd[:, -1] == 0))
        print("##########")

        model_list = []

        for i, _ in enumerate(tremor_model_list):
            tremor_model = tremor_model_list[i]
            rigidity_model = rigidity_model_list[i]

            freeze_model(tremor_model)
            freeze_model(rigidity_model)

            combined_logit_model = ModelFusionMultiLabel(
                tremor_model.logit.g, rigidity_model.logit.g, embedding_dim)
            model = MultiLabelModel2M(model=combined_logit_model)

            model.fit(train_loader, test_loader=train_loader,
                      num_epochs=finetuning_epochs)
            model_path = fused_models_path + "model_fused_" + str(i)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
            }, model_path)
            model_list.append(model)
    else:
        print("Loading pretrained fused models from folder")
        domain = imu_params["domain"]
        bag_length = imu_params["bag_length"]
        pooling = train_params["pooling"]
        embedding_dim = train_params["embedding_dim"]
        ht_split_threshold_ms = typing_params["ht_split_threshold_ms"]
        ft_split_threshold_ms = typing_params["ft_split_threshold_ms"]
        bin_width_ms = typing_params["bin_width_ms"]
        num_bins_ht = len(np.arange(0, ht_split_threshold_ms+1, bin_width_ms))
        num_bins_ft = len(np.arange(0, ft_split_threshold_ms+1, bin_width_ms))
        num_bins = num_bins_ht + num_bins_ft
        pretrained_model_list = glob.glob(fused_models_path + "/*")
        model_list = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i, model_path in enumerate(pretrained_model_list):
            if domain == "frequency":
                tremor_model = TremorMILFrequency(
                    M=embedding_dim, L=16, K=bag_length, pooling=pooling)
            elif domain == "time":
                tremor_model = TremorMILTime(
                    M=embedding_dim, L=16, K=bag_length, pooling=pooling)

            fmi_model = RigidityMIL(input_dim=num_bins, M=embedding_dim,
                                    L=16, K=bag_length)

            tremor_checkpoint = torch.load(pretrained_models_path + "model_tremor_" + str(i))
            fmi_checkpoint = torch.load(pretrained_models_path + "model_typing_" + str(i))
            tremor_model.load_state_dict(
                    tremor_checkpoint['model_state_dict'], strict=False)
            fmi_model.load_state_dict(
                    fmi_checkpoint['model_state_dict'], strict=False)
            tremor_model.to(device)
            fmi_model.to(device)

            combined_logit_model = ModelFusionMultiLabel(
                tremor_model.g, fmi_model.g, embedding_dim)
            model = MultiLabelModel2M(model=combined_logit_model)
            checkpoint = torch.load(model_path)
            model.load_state_dict(
                checkpoint['model_state_dict'], strict=False)
            model_list.append(model)


    print("#### Stage 3 - Evaluate PD predictions on GData ####")
    controlled_serials = get_controlled_serials()

    X_imu_gdata, X_typing_gdata, X_masks_imu_gdata, X_masks_typing_gdata, y_pd_gdata, serials_gdata = get_common_gdata_subset(
        imu_gdataset,
        typing_gdataset,
        imu_params,
        typing_params,
        serials_to_exclude=controlled_serials)

    y_pd_gdata = np.column_stack((y_pd_gdata, y_pd_gdata, y_pd_gdata))

    test_set = CustomDataset2M(
        X_imu_gdata, X_typing_gdata, y_pd_gdata,
        X_masks_imu_gdata, X_masks_typing_gdata, serials_gdata)

    test_loader = DataLoader(
        test_set, batch_size=8, shuffle=False)

    auc_score = evaluate_ensemble(model_list, test_loader, decision_th)
    print("")

    return auc_score

@ex.main
def deep_pd_mil(train_params, imu_params, typing_params,
                imu_gdataset, typing_gdataset,
                imu_sdataset, typing_sdataset,
                _run=None):
    evaluation_on = train_params["evaluation_on"]

    assert imu_params["domain"] in ["time", "frequency"]

    if evaluation_on == "sdata":
        result = evaluate_loso_sdata(train_params, imu_params, typing_params,
                                     imu_sdataset, typing_sdataset)

    elif evaluation_on == "gdata":
        result = evaluate_pretrained_gdata(train_params, imu_params,
                                           typing_params, imu_sdataset,
                                           typing_sdataset, imu_gdataset,
                                           typing_gdataset)

    return result


ex.run_commandline()
