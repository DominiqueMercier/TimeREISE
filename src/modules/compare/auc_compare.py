import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from modules.networks import model_utils
from sklearn.metrics import auc as auc_sklearn
from sklearn.metrics import classification_report

sns.set()


def create_auc_input(x, attr, mode='del'):
    if mode == 'del':
        x_tmp = x.copy()
    else:
        x_tmp = np.zeros(x.shape)
    queue = np.dstack(np.unravel_index(np.argsort(
        attr, axis=None), attr.shape))[0][::-1]
    masked_x = [x_tmp.copy()]
    for idx in range(queue.shape[0]):
        c, t = queue[idx]
        x_tmp[c, t] = 0 if mode == 'del' else x[c, t]
        masked_x.append(x_tmp.copy())
    masked_x = np.array(masked_x)
    return masked_x


def create_blank_report(methods):
    auc_report = {'scores': {}, 'summary': {}}
    for l1 in list(auc_report):
        auc_report[l1] = {'del': {}, 'ins': {}}
        for l2 in list(auc_report[l1]):
            auc_report[l1][l2] = {
                'accuracy': {}, 'macro avg/f1-score': {},
                'weighted avg/f1-score': {}}
            for l3 in list(auc_report[l1][l2]):
                for method in methods:
                    auc_report[l1][l2][l3][method] = []
    return auc_report


def insert_scores(auc_report, method, mode, preds, y):
    for i in range(preds.shape[0]):
        tmp_auc = classification_report(
            y, preds[i], digits=4, output_dict=True, zero_division=0)
        for metric in list(auc_report['scores'][mode]):
            score = tmp_auc[metric] if '/' not in metric else \
                tmp_auc[metric.split('/')[0]][metric.split('/')[1]]
            auc_report['scores'][mode][metric][method].append(score)


def insert_auc(auc_report, method, mode):
    for metric in list(auc_report['scores'][mode]):
        scores = auc_report['scores'][mode][metric][method]
        base = np.linspace(0, 1, len(scores))
        if mode == 'del':
            base = base[::-1]
        auc = auc_sklearn(base, scores)
        auc_report['summary'][mode][metric][method] = auc


def compute_auc(model, attrProcessor, x, y, batch_size=32):
    methods = [m for m in sorted(attrProcessor.approaches)
               if 'attr' in attrProcessor.approaches[m]]
    auc_report = create_blank_report(methods)
    for mode in list(auc_report['scores']):
        for attr_name in methods:
            attr = attrProcessor.get_attribution(attr_name)
            preds = []
            for idx, attr_map in enumerate(attr):
                auc_inputs = create_auc_input(x[idx], attr_map, mode)
                tmp_preds = model_utils.predict(model, auc_inputs,
                                                batch_size).numpy()
                preds.append(tmp_preds)
            preds = np.array(preds).T
            insert_scores(auc_report, attr_name, mode, preds, y)
            insert_auc(auc_report, attr_name, mode)
    return auc_report


def plot_auc(auc_report, not_show=False, save_path=None):
    modes = sorted(auc_report['scores'])
    metrics = sorted(auc_report['scores']['del'])
    methods = sorted(auc_report['scores']['del']['accuracy'])
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(20, 6), nrows=len(modes),
                               ncols=len(methods))
        axes = ax.flat
        pidx = 0
        for mode in modes:
            for method in methods:
                axes[pidx].set_title(method)
                scores = auc_report['scores'][mode][metric][method]
                auc = auc_report['summary'][mode][metric][method]
                indices = np.linspace(0, 1, len(scores))
                axes[pidx].fill_between(indices, scores, alpha=0.4)
                axes[pidx].plot(indices, scores, label='AUC: %.4f' % auc)
                axes[pidx].set_xlabel('data ' + mode + 'in [%]')
                axes[pidx].set_ylabel(metric)
                axes[pidx].legend()
                pidx += 1

        fig.tight_layout()

        if save_path is not None:
            fname = 'AUC_' + metric.replace('/', ' ') + '.png'
            plt.savefig(os.path.join(save_path, fname), dpi=300,
                        bbox_inches='tight', pad_inches=0.1)

        if not not_show:
            plt.show()
