import glob
import math
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import AttentionMILRigidity as RigidityMIL
from vat_pytorch.torch_utils import CustomDataset1M
from vat_pytorch.vat import Model1M

print = partial(print, flush=True)


def maybe_train_typing_models(typing_sdataset, typing_params,
                              train_params, serials_to_exclude=[],
                              left_out=None):
    X_typing, X_masks_typing, y_typing, serials_typing = prepare_typing_sdata(
        typing_sdataset, typing_params, serials_to_exclude)

    checkpoint = typing_params["checkpoint"]
    bag_length = typing_params["bag_length"]
    embedding_dim = train_params["embedding_dim"]
    pooling = train_params["pooling"]

    ht_split_threshold_ms = typing_params["ht_split_threshold_ms"]
    ft_split_threshold_ms = typing_params["ft_split_threshold_ms"]
    bin_width_ms = typing_params["bin_width_ms"]
    num_bins_ht = len(np.arange(0, ht_split_threshold_ms+1, bin_width_ms))
    num_bins_ft = len(np.arange(0, ft_split_threshold_ms+1, bin_width_ms))
    num_bins = num_bins_ht + num_bins_ft

    model_list = []

    if checkpoint is False:
        batch_size = train_params["batch_size"]
        num_trials = train_params["num_trials"]
        pre_training_epochs = train_params["pre_training_epochs"]

        training_set = CustomDataset1M(
            X_typing, y_typing, X_masks_typing, serials_typing)

        train_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True)

        print("#### Pre-training typing detection model ####")
        print("Total subjects: ", y_typing.shape[0])
        print("Positive subjects: ", np.count_nonzero(y_typing == 1))
        print("Healthy subjects: ", np.count_nonzero(y_typing == 0))
        print("")

        for t in range(num_trials):
            logit_model = RigidityMIL(input_dim=num_bins, M=embedding_dim,
                                      L=16, K=bag_length)

            model = Model1M(model=logit_model)

            _, model_path = model.fit(train_loader,
                                      test_loader=train_loader,
                                      mode="cr", num_epochs=4*pre_training_epochs)
            model_list.append(model)

            del model
            del logit_model

    else:
        print("#### Loading Typing models from pretrained folder ####")

        if left_out is not None:
            pretrained_models_path = "models/pretrained/sdata/"
            pretrained_model_list = glob.glob(pretrained_models_path + "/*")

            for model_path in pretrained_model_list:
                model_name = model_path.split("/")[-1]

                if left_out in model_name and "typing" in model_name:
                    logit_model = RigidityMIL(input_dim=num_bins, M=64,
                                              L=16, K=bag_length)
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

                if "typing" in model_name:
                    logit_model = RigidityMIL(input_dim=num_bins, M=64,
                                              L=16, K=bag_length)
                    checkpoint = torch.load(model_path)
                    model = Model1M(model=logit_model)
                    model.logit.load_state_dict(
                        checkpoint['model_state_dict'], strict=False)
                    model_list.append(model)

    return model_list


def is_typing_session_valid(dt, ut, min_keystrokes):
    if len(ut) != len(dt):
        return False

    if len(ut) < min_keystrokes:
        return False

    return True


def compute_typing_session_histograms(doc,
                                      ft_split_threshold_ms,
                                      ht_split_threshold_ms,
                                      min_keystrokes_per_session,
                                      bin_width_ms):
    payload = doc['payload']
    dt = np.array(payload['DownTime'])
    ut = np.array(payload['UpTime'])

    if not is_typing_session_valid(dt, ut, min_keystrokes_per_session):
        return None, None

    #is_long_press = payload['IsLongPress']

    norm_dt = dt - dt[0]
    norm_ut = ut - dt[0]

    if norm_ut[-1] <= norm_dt[-1]:
        print("Corrupted session")

        return None, None

    all_together = np.sort(np.concatenate((norm_dt, norm_ut)))
    flight_times = np.diff(all_together[1:])[::2]
    hold_times = np.diff(all_together)[::2]
    hold_times = hold_times[:-1]

    if len(flight_times) < min_keystrokes_per_session:
        return None, None

    ht_bins = np.arange(0, ht_split_threshold_ms+1, bin_width_ms)
    ht_bins = np.append(ht_bins, np.inf)
    ht_hist, ht_edges = np.histogram(hold_times, bins=ht_bins)
    ht_hist = ht_hist / np.sum(ht_hist)

    ft_bins = np.arange(0, ft_split_threshold_ms+1, bin_width_ms)
    ft_bins = np.append(ft_bins, np.inf)
    ft_hist, ft_edges = np.histogram(flight_times, bins=ft_bins)
    ft_hist = ft_hist / np.sum(ft_hist)

    session_features = np.concatenate((ht_hist, ft_hist))

    return session_features, len(flight_times)


def make_typing_bags(sequences, bag_length):
    bags = []
    masks = []

    #### Zero-prepad ####
    Ws = sequences.shape[1]
    num_sequences = sequences.shape[0]

    nearest_bag_length_multiple = math.ceil(
        num_sequences/bag_length)*bag_length
    padding_length = nearest_bag_length_multiple - num_sequences

    sequences = np.concatenate(
        (np.zeros((padding_length, Ws)), sequences))

    ##### MASK STUFF #####
    this_subject_mask = np.ones(sequences.shape[0])
    this_subject_mask[:padding_length] = 0

    this_num_bags = nearest_bag_length_multiple // bag_length
    this_bags = np.split(sequences, this_num_bags, axis=0)
    this_bags = np.array(this_bags)
    this_bag_masks = np.split(this_subject_mask, this_num_bags, axis=0)
    this_bag_masks = np.array(this_bag_masks)

    bags.append(this_bags)
    masks.append(this_bag_masks)

    bags = np.concatenate(bags, axis=0)
    masks = np.concatenate(masks, axis=0)
    num_bags = bags.shape[0]

    return bags, masks, num_bags


def prepare_typing_devdata(typing_params):
    ft_split_threshold_ms = typing_params["ft_split_threshold_ms"]
    ht_split_threshold_ms = typing_params["ht_split_threshold_ms"]
    min_keystrokes_per_session = typing_params["min_keystrokes_per_session"]
    target = typing_params["target"]
    bag_length = typing_params["bag_length"]
    min_sessions = typing_params["min_sessions"]
    bin_width_ms = typing_params["bin_width_ms"]

    sessions_dict = load_typing_devset()
    target_serials = sessions_dict.keys()

    X = []
    X_masks = []
    serials = []
    y = []

    for serial in target_serials:
        this_sessions_raw = sessions_dict[serial][0]
        updrs22 = sessions_dict[serial][1]
        updrs23 = sessions_dict[serial][2]
        pd_status = sessions_dict[serial][-1]

        if target == "updrs22":
            ground_truth = updrs22
        elif target == "updrs23":
            ground_truth = updrs23
        elif target == "fmi":
            ground_truth = int(np.logical_and(updrs22, updrs23))

        ground_truth = [ground_truth]

        if len(this_sessions_raw) < min_sessions:
            continue

        this_sessions = []
        this_session_keystrokes = []

        for session in this_sessions_raw:
            doc = {}
            doc['payload'] = {}
            doc['payload']['DownTime'] = session[:, 0]
            doc['payload']['UpTime'] = session[:, 1]
            typing_sessions, session_keystrokes = compute_typing_session_histograms(doc,
                                                                                    ft_split_threshold_ms,
                                                                                    ht_split_threshold_ms,
                                                                                    min_keystrokes_per_session, bin_width_ms)

            if typing_sessions is None:
                continue
            this_sessions.append(typing_sessions)
            this_session_keystrokes.append(session_keystrokes)

        # Sort sessions based on number of keystrokes
        this_session_keystrokes = np.array(this_session_keystrokes)
        idx = np.argsort(this_session_keystrokes)[::-1]
        this_sessions = [this_sessions[i] for i in idx]
        this_session_keystrokes = this_session_keystrokes[idx]

        this_sessions = this_sessions[:bag_length]
        this_sessions = np.array(this_sessions)

        # Call make_bags to get padding on the instance level as well.
        this_bags_train, this_bag_masks_train, this_num_bags_train = make_typing_bags(
            this_sessions, bag_length)
        X.append(this_bags_train)
        X_masks.append(this_bag_masks_train)
        serials.append([str(serial)] * this_num_bags_train)
        #y.append([ground_truth] * this_num_bags_train)
        y.append(ground_truth * this_num_bags_train)

    y = np.concatenate(y)
    y[y > 0] = 1
    serials = np.concatenate(serials)
    X_masks = np.concatenate(X_masks, axis=0).astype('float32')
    X = np.concatenate(X, axis=0).astype('float32')

    return X, X_masks, y, serials


def prepare_typing_sdata(dataset, typing_params, serials_to_exclude=[]):
    target = typing_params["target"]
    bag_length = typing_params["bag_length"]
    min_sessions = typing_params["min_sessions"]

    dataset = "data/" + dataset
    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))

    sessions_dict = pickle.load(open(dataset, "rb"))
    target_serials = sessions_dict.keys()

    X = []
    X_masks = []
    serials = []
    y = []

    for serial in target_serials:
        if serial in serials_to_exclude:
            continue

        user_doc = subject_metadata[serial]
        med_eval_1 = user_doc["med_eval_1"]
        med_eval_2 = user_doc["med_eval_2"]
        pd_status = med_eval_1["pd_status"]
        updrs_22_1 = med_eval_1["updrs_22_uextr_right"] + \
            med_eval_1["updrs_22_uextr_left"]
        updrs_22_2 = med_eval_2["updrs_22_uextr_right"] + \
            med_eval_2["updrs_22_uextr_left"]
        updrs_23_1 = med_eval_1["updrs_23_right"] + med_eval_1["updrs_23_left"]
        updrs_23_2 = med_eval_2["updrs_23_right"] + med_eval_2["updrs_23_left"]

        fpp_fmi = med_eval_1["fpp_ams_fine_motor"] + \
            med_eval_2["fpp_ams_fine_motor"]

        updrs22_total = updrs_22_1 + updrs_22_2
        updrs23_total = updrs_23_1 + updrs_23_2

        if target == "updrs22":
            ground_truth = updrs22_total
        elif target == "updrs23":
            ground_truth = updrs23_total
        elif target == "fmi":
            if pd_status == 0:
                ground_truth = 0
            else:
                ground_truth = int(np.logical_or(np.logical_and(
                    updrs22_total, updrs23_total), fpp_fmi))

        this_sessions = sessions_dict[serial][0]

        if len(this_sessions) < min_sessions:
            continue

        this_sessions = this_sessions[:bag_length]
        this_sessions = np.array(this_sessions)

        # Call make_bags to get padding on the instance level as well.
        this_bags_train, this_bag_masks_train, this_num_bags_train = make_typing_bags(
            this_sessions, bag_length)
        X.append(this_bags_train)
        X_masks.append(this_bag_masks_train)
        serials.append([serial] * this_num_bags_train)
        y.append([ground_truth] * this_num_bags_train)

    y = np.concatenate(y)
    y[y > 0] = 1
    serials = np.concatenate(serials)
    X_masks = np.concatenate(X_masks, axis=0).astype('float32')
    X = np.concatenate(X, axis=0).astype('float32')

    if typing_params["use_development_set"] is True:
        # Incorporate development set
        X_dev, X_masks_dev, y_dev, serials_dev = prepare_typing_devdata(
            typing_params)
        X = np.concatenate((X, X_dev))
        X_masks = np.concatenate((X_masks, X_masks_dev))
        y = np.concatenate((y, y_dev))
        serials = np.concatenate((serials, serials_dev))

    return X, X_masks, y, serials


def prepare_typing_gdata(dataset, typing_params, serials_to_exclude=[]):
    bag_length = typing_params["bag_length"]
    min_sessions = typing_params["min_sessions"]
    dataset = "data/" + dataset
    subject_metadata = pickle.load(open("data/subject_metadata.pickle", "rb"))

    sessions_dict = pickle.load(open(dataset, "rb"))
    target_serials = sessions_dict.keys()

    X = []
    X_masks = []
    serials = []
    y = []

    for serial in target_serials:
        pd_status = sessions_dict[serial][-1]

        this_sessions = sessions_dict[serial][0]

        if len(this_sessions) < min_sessions:
            continue

        this_sessions = this_sessions[:bag_length]
        this_sessions = np.array(this_sessions)

        # Call make_bags to get padding on the instance level as well.
        this_bags_train, this_bag_masks_train, this_num_bags_train = make_typing_bags(
            this_sessions, bag_length)
        X.append(this_bags_train)
        X_masks.append(this_bag_masks_train)
        serials.append([serial] * this_num_bags_train)
        y.append([pd_status] * this_num_bags_train)

    y = np.array(y)
    y[y > 0] = 1
    serials = np.concatenate(serials)
    X_masks = np.concatenate(X_masks, axis=0).astype('float32')
    X = np.concatenate(X, axis=0).astype('float32')

    return X, X_masks, y, serials


def get_clinical_characteristics_for_subject(subject_id):
    df = pd.read_csv(
        'data/iprognosis_typing_dev/Demographics_Clinical_Characteristics.csv')
    break_flag = False

    for index, row in df.iterrows():
        if row['Subject ID'] == subject_id:
            break_flag = True

            break

    if not break_flag:
        raise "ERROR"

    return row


def load_typing_devset():
    data_root = "data/iprognosis_typing_dev/Data"
    subject_folders = glob.glob(data_root + "/*")

    subject_serials = [int(i[-2:]) for i in subject_folders]

    subject_dict = {}

    for j, subject_folder in enumerate(subject_folders):
        # print(j)
        subject_id = subject_serials[j]
        this_subject_sessions_files = glob.glob(subject_folder + "/*.txt")

        subject_sessions = []

        for session_file in this_subject_sessions_files:
            #print(subject_id, session_file)
            with open(session_file, "r") as fh:
                content = fh.readlines()
            content = [x.strip() for i, x in enumerate(content) if i > 0]

            down = []
            up = []

            for line in content:
                line_content = line.split(',')

                if len(line_content) == 1:
                    continue
                press = int(line_content[1])
                release = int(line_content[2].split(' ')[1])
                down.append(press)
                up.append(release)

            down = np.array(down)
            up = np.array(up)
            subject_sessions.append(np.column_stack((down, up)))

        updrs = get_clinical_characteristics_for_subject(subject_id)
        updrs22 = int(updrs['UPDRS_III Item 22 Rigidity-Right hand']) \
            + int(updrs['UPDRS_III Item 22 Rigidity-Left hand'])
        updrs23 = int(updrs['UPDRS_III Item 23 Finger Taps-Right hand']) + \
            int(updrs['UPDRS_III Item 23 Finger Taps-Left hand'])
        pd_status = 0

        if updrs['Group'] != "Control":
            pd_status = 1

        subject_dict[subject_id] = [
            subject_sessions, updrs22, updrs23, pd_status]

    return subject_dict
