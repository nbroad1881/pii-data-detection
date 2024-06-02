import pickle

import numpy as np
import lightgbm as lgb


def lgb_fbeta_score(y_hat, data):
    y_true = data.get_label()
    y_hat = y_hat.argmax(-1)

    FP = len([1 for gt, p in zip(y_true, y_hat) if gt == label2id["O"] and p != label2id["O"]])
    TP = len([1 for gt, p in zip(y_true, y_hat) if gt != label2id["O"] and p == gt])
    FN = len([1 for gt, p in zip(y_true, y_hat) if gt != label2id["O"] and p == label2id["O"]])


    beta = 5
    s_micro = (1 + (beta**2)) * TP / (((1 + (beta**2)) * TP) + ((beta**2) * FN) + FP)

    return "f5", s_micro, True


with open("/drive2/kaggle/pii-dd/data/short_oof_v1.pkl", "rb") as f:
    features, labels = pickle.load(f)

unique_labels = sorted(list(set(labels)))
label2id = {label: i for i, label in enumerate(unique_labels)}

label_ids = np.array([label2id[label] for label in labels])
data = np.array(features)


train_data = lgb.Dataset(data, label=label_ids)

param = {
    "num_leaves": 30,
    "max_depth": 30,
    # "learning_rate": 0.3,
    "objective": "multiclass",
    "num_class": len(unique_labels),
    "device_type": "gpu",
}

evals_result = {}
num_round = 50

results = lgb.cv(
    param,
    train_data,
    num_round,
    nfold=4,
    feval=lgb_fbeta_score,
)

print(max(results["valid f5-mean"]))
