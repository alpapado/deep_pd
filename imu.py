import glob
import math
import pickle
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import AttentionMILTremorFrequency as TremorMILFrequency
from models import AttentionMILTremorTime as TremorMILTime
from vat_pytorch.torch_utils import CustomDataset1M
from vat_pytorch.vat import Model1M

print = partial(print, flush=True)

def maybe_train_imu_models(imu_sdataset, imu_params,
                           train_params, serials_to_exclude=[],
                           left_out=None):
    X_imu, X_masks_imu, y_imu, serials_imu = prepare_imu_sdata(
        imu_sdataset, imu_params, serials_to_exclude)

    checkpoint = imu_params["checkpoint"]
    domain = imu_params["domain"]
    bag_length = imu_params["bag_length"]
    pooling = train_params["pooling"]
    embedding_dim = train_params["embedding_dim"]

    model_list = []

    if checkpoint is False:
        batch_size = train_params["batch_size"]
        num_trials = train_params["num_trials"]
        pre_training_epochs = train_params["pre_training_epochs"]

        training_set = CustomDataset1M(X_imu, y_imu, X_masks_imu, serials_imu)

        train_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True)

        print("#### Pre-training tremor detection model ####")
        print("Total subjects: ", y_imu.shape[0])
        print("Tremorous subjects: ", np.count_nonzero(y_imu == 1))
        print("Healthy subjects: ", np.count_nonzero(y_imu == 0))
        print("")

        for t in range(num_trials):
            if domain == "frequency":
                logit_model = TremorMILFrequency(
                    M=embedding_dim, L=16, K=bag_length, pooling=pooling)
            elif domain == "time":
                logit_model = TremorMILTime(
                    M=embedding_dim, L=16, K=bag_length, pooling=pooling)

            model = Model1M(model=logit_model)

            _, model_path = model.fit(train_loader,
                                      test_loader=train_loader,
                                      mode="cr", num_epochs=pre_training_epochs)
            model_list.append(model)

            del model
            del logit_model
    else:
        print("#### Loading IMU models from pretrained folder ####")
        if left_out is not None:
            pretrained_models_path = "models/pretrained/sdata/"
            pretrained_model_list = glob.glob(pretrained_models_path + "/*")
            for model_path in pretrained_model_list:
                model_name = model_path.split("/")[-1]
                if left_out in model_name and "tremor" in model_name:
                    if domain == "frequency":
                        logit_model = TremorMILFrequency(
                            M=embedding_dim, L=16, K=bag_length, pooling=pooling)
                    elif domain == "time":
                        logit_model = TremorMILTime(
                            M=embedding_dim, L=16, K=bag_length, pooling=pooling)

                    checkpoint = torch.load(model_path)
                    model = Model1M(model=logit_model)
                    model.logit.load_state_dict(
                        checkpoint['model_state_dict'], strict=False)
                    model_list.append(model)
        else:
            pretrained_models_path = "models/pretrained/gdata/"
            pretrained_model_list = glob.glob(pretrained_models_path + "/*")
            for model_path in pretrained_model_list:
                model_name = model_path.split("/")[-1]
                if "tremor" in model_name:
                    if domain == "frequency":
                        logit_model = TremorMILFrequency(
                            M=embedding_dim, L=16, K=bag_length, pooling=pooling)
                    elif domain == "time":
                        logit_model = TremorMILTime(
                            M=embedding_dim, L=16, K=bag_length, pooling=pooling)

                    checkpoint = torch.load(model_path)
                    model = Model1M(model=logit_model)
                    model.logit.load_state_dict(
                        checkpoint['model_state_dict'], strict=False)
                    model_list.append(model)

    return model_list


def make_imu_bags(this_windows, bag_length):
    bags = []
    masks = []

    #### Zero-prepad ####
    Ws = this_windows.shape[1]
    C = this_windows.shape[2]
    num_windows = this_windows.shape[0]

    nearest_bag_length_multiple = math.ceil(
        num_windows/bag_length)*bag_length
    padding_length = nearest_bag_length_multiple - num_windows

    this_windows = np.concatenate(
        (np.zeros((padding_length, Ws, C)), this_windows))

    ##### MASK STUFF #####
    this_subject_mask = np.ones(this_windows.shape[0])
    this_subject_mask[:padding_length] = 0

    this_num_bags = nearest_bag_length_multiple // bag_length
    this_bags = np.split(this_windows, this_num_bags, axis=0)
    this_bags = np.array(this_bags)
    this_bag_masks = np.split(this_subject_mask, this_num_bags, axis=0)
    this_bag_masks = np.array(this_bag_masks)

    bags.append(this_bags)
    masks.append(this_bag_masks)

    bags = np.concatenate(bags, axis=0)
    masks = np.concatenate(masks, axis=0)
    num_bags = bags.shape[0]

    return bags, masks, num_bags


def prepare_imu_gdata(dataset, imu_params, serials_to_exclude=[]):
    bag_length = imu_params["bag_length"]
    min_windows = imu_params["min_windows"]
    domain = imu_params["domain"]

    sessions_dict = pickle.load(open("data/"+dataset, "rb"))
    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))
    serials = sessions_dict.keys()

    X_train = []
    X_train_masks = []
    X_train_serials = []
    y_train = []

    for serial in serials:
        if serial in serials_to_exclude:
            continue

        user_doc = subject_metadata[serial]
        if 2020 - user_doc["age"] > 80:
            continue

        if domain == "frequency":
            this_sessions = sessions_dict[serial][0]
        elif domain == "time":
            this_sessions = sessions_dict[serial][3]

        this_windows = []
        this_window_energies = []

        for i, sess in enumerate(this_sessions):
            for window in sess:
                this_windows.append(window)
                this_window_energies.append(np.sum(window))

        this_windows = np.array(this_windows)
        this_window_energies = np.array(this_window_energies)

        if this_windows.shape[0] < min_windows:
            continue

        window_energy_in_pd_tremor_band = np.sum(
            this_windows.sum(axis=2)[:, 9:22], axis=1)
        sort_idx = np.argsort(window_energy_in_pd_tremor_band)[::-1]
        this_windows = this_windows[sort_idx][:bag_length]

        window_energy_in_pd_tremor_band = window_energy_in_pd_tremor_band[
            sort_idx][:bag_length]

        this_train_windows = this_windows
        ground_truth = -1

        this_bags_train, this_bag_masks_train, this_num_bags_train = make_imu_bags(
            this_train_windows, bag_length)

        X_train.append(this_bags_train.astype('float32'))
        X_train_masks.append(this_bag_masks_train)
        X_train_serials.append([serial] * this_num_bags_train)
        y_train.append([ground_truth] * this_num_bags_train)

    y_train = np.concatenate(y_train, axis=0)
    y_train[y_train > 0] = 1
    X_train_masks = np.concatenate(X_train_masks, axis=0).astype('float32')
    X_train = np.concatenate(X_train, axis=0).astype('float32')
    X_train_serials = np.concatenate(X_train_serials, axis=0)
    X_train = np.swapaxes(X_train, -1, -2)

    return X_train, X_train_masks, y_train, X_train_serials


def prepare_imu_sdata(dataset, imu_params, serials_to_exclude=[]):
    bag_length = imu_params["bag_length"]
    min_windows = imu_params["min_windows"]
    domain = imu_params["domain"]

    sessions_dict = pickle.load(open("data/"+dataset, "rb"))
    target_serials = sessions_dict.keys()

    X = []
    X_masks = []
    serials = []
    y = []

    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))
    for serial in target_serials:
        if serial in serials_to_exclude:
            continue
        subject_updrs = sessions_dict[serial][1]

        if "fpp_ams_tremor" not in subject_updrs:
            continue

        user_doc = subject_metadata[serial]
        ground_truth = user_doc["med_eval_1"]["tremor_manual_2_2_2020_new"]

        if domain == "frequency":
            this_sessions = sessions_dict[serial][0]
        elif domain == "time":
            this_sessions = sessions_dict[serial][3]

        this_windows = []
        this_window_energies = []

        for i, sess in enumerate(this_sessions):
            for window in sess:
                this_windows.append(window)
                this_window_energies.append(np.sum(window))

        this_windows = np.array(this_windows)
        this_window_energies = np.array(this_window_energies)

        if this_windows.shape[0] < min_windows:
            continue

        window_energy_in_pd_tremor_band = np.sum(
            this_windows.sum(axis=2)[:, 9:22], axis=1)
        sort_idx = np.argsort(window_energy_in_pd_tremor_band)[::-1]
        this_windows = this_windows[sort_idx][:bag_length]

        window_energy_in_pd_tremor_band = window_energy_in_pd_tremor_band[
            sort_idx][:bag_length]

        this_bags, this_bag_masks, this_num_bags = make_imu_bags(
            this_windows, bag_length)

        X.append(this_bags)
        X_masks.append(this_bag_masks)
        serials.append([serial] * this_num_bags)
        y.append([ground_truth] * this_num_bags)

    y = np.concatenate(y, axis=0)
    y[y > 0] = 1
    X_masks = np.concatenate(X_masks, axis=0).astype('float32')
    X = np.concatenate(X, axis=0).astype('float32')
    serials = np.concatenate(serials, axis=0)
    X = np.swapaxes(X, -1, -2)

    return X, X_masks, y, serials
