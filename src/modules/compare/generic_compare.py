import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from captum.metrics import infidelity, sensitivity_max

sns.set()


def compute_infidelity(attrProcessor, data, target, name, scale=0.05,
                       n_perturb_samples=1000, batch_size=32):
    device = attrProcessor.device
    data_tensor = torch.Tensor(data)
    labels = torch.LongTensor(target)
    attr_tensor = torch.Tensor(
        attrProcessor.get_attribution(name))
    rng = np.random.RandomState(0)

    def perturb_fn(inputs):
        noise = torch.Tensor(rng.uniform(
            -scale, scale, inputs.size())).to(device)
        return noise, inputs + noise

    infidelities = []
    for i in range(0, data_tensor.size()[0], batch_size):
        d, t = data_tensor[i:i+batch_size], labels[i:i+batch_size]
        a, b = attr_tensor[i:i+batch_size], d.size(0)
        infid = infidelity(
            attrProcessor.model, perturb_fn, d.to(device), a.to(device),
            n_perturb_samples=n_perturb_samples, max_examples_per_batch=b,
            normalize=True, target=t.to(device))
        infid = infid.detach().to('cpu').numpy()
        infidelities.append(infid)
    infidelities = np.nan_to_num(np.concatenate(infidelities))
    return infidelities


def compute_sensitivity(attrProcessor, data, target, name, scale=0.02,
                        n_perturb_samples=10, batch_size=32):
    device = attrProcessor.device
    rng = np.random.RandomState(0)
    data_tensor = torch.Tensor(data)
    labels = torch.LongTensor(target)

    def perturb_fn(inputs):
        noise = torch.Tensor(rng.uniform(
            -scale, scale, inputs.size())).to(device)
        perturbed_input = inputs + noise
        return tuple([perturbed_input])

    def explanation_func(x, y):
        attr = attrProcessor.approaches[name]['method'].attribute(x[0], y)
        return (attr,)

    sensitivies = []
    for i in range(0, data.shape[0]):
        d, t = data_tensor[i:i+1], labels[i:i+1]
        sens = sensitivity_max(
            explanation_func, d.to(device), perturb_func=perturb_fn,
            n_perturb_samples=n_perturb_samples,
            max_examples_per_batch=d.size()[0], y=t.to(device))
        sens = sens.detach().to('cpu').numpy()
        sensitivies.append(sens)
    sensitivies = np.nan_to_num(np.concatenate(sensitivies))
    return sensitivies


def compute_inf_sens(attrProcessor, data, target, scale,
                     n_perturb_samples, mode, batch_size=32):
    approaches = sorted(attrProcessor.approaches)
    if mode == 'Infidelity':
        compute_func = compute_infidelity
        methods = [m for m in approaches if 'attr'
                   in attrProcessor.approaches[m]]
    else:
        compute_func = compute_sensitivity
        methods = [m for m in approaches if
                   attrProcessor.attr_config[m]['execute']]
    report = {'scores': {}, 'summary': {}}
    for name in methods:
        scores = compute_func(attrProcessor, data, target, name, scale,
                              n_perturb_samples, batch_size)
        report['scores'][name] = scores
        s, m = np.std(scores), np.mean(scores)
        report['summary'][name] = {'mean': s, 'std': m}
    return report


def compute_continuity(attrProcessor):
    approaches = sorted(attrProcessor.approaches)
    methods = [m for m in approaches if 'attr'
               in attrProcessor.approaches[m]]
    report = {'scores': {}, 'summary': {}}
    for name in methods:
        attrs = attrProcessor.get_attribution(name)
        scores = np.absolute(attrs[:, :, :-1] - attrs[:, :, 1:])
        report['scores'][name] = scores
        s, m = np.std(scores), np.mean(scores)
        report['summary'][name] = {'mean': s, 'std': m}
    return report


def plot(report, mode, scale=None, n_perturb_samples=None, not_show=False,
         save_path=None):
    methods = sorted(report['scores'])
    metrics = report['summary'][methods[0]]
    n_bars = np.arange(len(report['scores']))
    fig, ax = plt.subplots(figsize=(20, 3), ncols=len(metrics))
    for i, m in enumerate(sorted(metrics)):
        inf = np.array([report['summary'][method][m] for method in methods])
        ax[i].set_title(mode + ': ' + m)
        ax[i].bar(n_bars, inf)
        ax[i].set_xlabel('Attribution method')
        ax[i].set_ylabel(mode)
        ax[i].set_xticks(n_bars, labels=methods)
        ax[i].set_ylim(np.min(inf) * 0.95)

    fig.tight_layout()

    if save_path is not None:
        fname = mode + '_%s_%s.png' % (scale, n_perturb_samples)
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()
