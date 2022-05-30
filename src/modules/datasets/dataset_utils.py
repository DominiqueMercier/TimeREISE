import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def scale_data(train_x, test_x, mode='standardize', frange=(0, 1)):
    if mode == 'standardize':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=frange)
    org_train_shape = train_x.shape[1:]
    org_test_shape = test_x.shape[1:]
    train_xf = train_x.reshape(-1, org_train_shape[-1])
    test_xf = test_x.reshape(-1, org_test_shape[-1])
    scaler.fit(train_xf)
    train_x = scaler.transform(train_xf).reshape(-1, *org_train_shape)
    test_x = scaler.transform(test_xf).reshape(-1, *org_test_shape)
    return train_x, test_x


def preprocess_data(train_x, train_y, test_x, test_y, normalize=False,
                    standardize=True, frange=(0, 1), channel_first=True):
    # adjust labels
    le = LabelEncoder().fit(train_y)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)
    # remove missings
    cmean = np.nanmean(train_x, axis=(0, 1))
    inds = np.where(np.isnan(train_x))
    train_x[inds] = np.take(cmean, inds[2])
    inds = np.where(np.isnan(test_x))
    test_x[inds] = np.take(cmean, inds[2])
    # standardize
    if standardize:
        train_x, test_x = scale_data(train_x, test_x, mode='standardize')
    # normalize
    if normalize:
        train_x, test_x = scale_data(
            train_x, test_x, mode='normalize', frange=frange)
    if channel_first:
        train_x = train_x.transpose(0, 2, 1)
        test_x = test_x.transpose(0, 2, 1)
    return train_x, train_y, test_x, test_y


def perform_datasplit(data, labels, test_split=0.3, stratify=True,
                      return_state=False, random_state=0):
    try:
        da, db, la, lb = train_test_split(
            data, labels, test_size=test_split, random_state=random_state,
            stratify=labels if stratify else None)
        state = True
    except:
        da, db, la, lb = train_test_split(
            data, labels, test_size=test_split, random_state=random_state,
            stratify=None)
        print('Warining: No stratified split possible')
        state = False
    if return_state:
        return da, la, db, lb, state
    return da, la, db, lb


def fuse_train_val(train_x, train_y, valX, valY):
    split_id = train_y.shape[0]
    train_x, train_y = np.stack([train_x, valX]), np.stack([train_y, valY])
    return train_x, train_y, split_id


def unfuse_train_val(data, labels, split_id):
    train_x, train_y = data[:split_id], labels[:split_id]
    valX, valY = data[split_id:], labels[split_id:]
    return train_x, train_y, valX, valY


def sub_sample(data, labels, ratio):
    sub_ids = np.random.permutation(np.arange(data.shape[0]))
    if ratio < 1:
        sub_ids = sub_ids[:int(ratio * data.shape[0])]
    else:
        sub_ids = sub_ids[:int(ratio)]
    sub_data, sub_labels = data[sub_ids], labels[sub_ids]
    return sub_data, sub_labels, sub_ids
