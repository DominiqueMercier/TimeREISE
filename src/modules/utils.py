import json
import os
import pickle

from sklearn.metrics import classification_report


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def compute_classification_report(gt, preds, save=False, store_dict=False,
                                  verbose=0):
    s = classification_report(gt, preds, digits=4)
    if verbose:
        print(s)
    if save is not None:
        with open(save, 'w') as f:
            f.write(s)
        if verbose:
            print('Save Location:', save)
        if store_dict:
            cr_dict = classification_report(
                gt, preds, digits=4, output_dict=True, zero_division=0)
            with open(save.replace('.txt', '.pickle'), 'wb') as f:
                pickle.dump(cr_dict, f)


def maybe_create_dirs(dataset_name, root='../../', dirs=['models', 'results'],
                      exp=None, return_paths=False, verbose=0):
    paths = []
    for d in dirs:
        if exp is None:
            tmp = os.path.join(root, d, dataset_name)
        else:
            tmp = os.path.join(root, d, exp, dataset_name)
        paths.append(tmp)
        if not os.path.exists(tmp):
            os.makedirs(tmp)
            if verbose:
                print('Created directory:', tmp)
        elif verbose:
            print('Found existing directory:', tmp)
    if return_paths:
        return paths


def get_pretty_dict(dictionary, sort=False, save=None, verbose=0):
    s =json.dumps(str(dictionary), sort_keys=sort, indent=4)
    if verbose:
        print(s)
    if save is not None:
        with open(save, 'w') as f:
            f.write(s)


def optin(verbose=True, file=None, **kwargs):
    pdict = dict(**kwargs)
    pstr = json.dumps(pdict)
    if verbose:
        print(pstr)
    if file is not None:
        with open(file, 'a') as f:
            f.write(pstr)
