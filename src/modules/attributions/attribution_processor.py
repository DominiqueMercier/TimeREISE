import os
from time import time

import matplotlib.pyplot as plt
import modules.attributions.captum_approaches as cp
import numpy as np
import seaborn as sns
import torch
from modules.attributions.timereise.perturbations import pert_funcs
from modules.attributions.timereise.timereise import Timereise
from modules.utils import get_pretty_dict

sns.set()


class ClassificationProcessor:
    def __init__(self, model, input_shape, attr_config={}, save_memory=False,
                 attr_dir=None, load=True, save=True, verbose=0):
        self.device = next(model.parameters()).device
        self.model = model
        self.input_shape = input_shape
        self.attr_config = attr_config
        self.approaches = {}
        self.save_memory = save_memory
        self.attr_dir = attr_dir
        self.load = load
        self.save = save
        self.verbose = verbose
        # read existing attr dict
        if len(attr_config) > 0:
            if self.verbose:
                print('Prepared: ', end='')
            for name in sorted(self.attr_config):
                self.init_approach(name)
                if self.verbose:
                    print('%s...' % name, end='')
            if self.verbose:
                print('Done')
        # load from folder
        if self.attr_dir is not None and self.load:
            names = [p.replace('_attr.npy', '')
                     for p in os.listdir(self.attr_dir) if '_attr.npy' in p]
            for name in names:
                self.load_attribution(name)

    def init_approach(self, name):
        if name == 'GuidedBackprop':
            self.approaches['GuidedBackprop'] = {
                'method': cp.GuidedBackprop(self.model)}

        elif name == 'IntegratedGradients':
            self.approaches['IntegratedGradients'] = {
                'method': cp.IntegratedGradients(self.model)}

        elif name == 'FeatureAblation':
            self.approaches['FeatureAblation'] = {
                'method': cp.FeatureAblation(self.model)}

        elif name == 'Occlusion':
            self.approaches['Occlusion'] = {
                'method': cp.Occlusion(self.model),
                'config': self.attr_config['Occlusion']['config']}

        elif name == 'Lime':
            self.approaches['Lime'] = {
                'method': cp.Lime(self.model),
                'config': self.attr_config['Lime']['config']}

        elif name == 'Timereise':
            # adjust mask path
            mask_path = None
            if self.attr_dir is not None:
                mask_path = os.path.join(self.attr_dir, 'timereise_masks.pt')
            self.approaches['Timereise'] = {
                'method': Timereise(self.model, self.input_shape,
                                    mask_path=mask_path,
                                    **self.attr_config['Timereise']
                                    ['config'])}
            # set correct default perturbation
            default_pert = pert_funcs[
                self.attr_config['Timereise']['config']['perturbation']](
                    **self.attr_config['Timereise']['pert_config'])
            self.approaches['Timereise']['method'].perturbation = default_pert

    def compute_all_attributions(self, data, target, folder=None):
        for name in sorted(list(self.approaches)):
            # methods that should not be executed
            if not self.attr_config[name]['execute']:
                continue
            # preprocess data and labels
            data_tensor = torch.Tensor(data).to(self.device)
            if len(data.shape) < 3:
                data_tensor = data_tensor.unsqueeze(0)
                t = int(target)
            else:
                t = [int(tar) for tar in target]
            self.compute_attributions(name, data_tensor, t)
            if self.verbose:
                print('Approach: %s | Time: %.3fs' %
                      (name, self.approaches[name]['time']))
            if folder is not None and self.save:
                attr_path = self.save_attribution(
                    name, folder, return_path=True)
                if self.save_memory:
                    self.approaches[name]['attr'] = attr_path
            if self.verbose:
                print('Finished method', name)
        if folder is not None:
            get_pretty_dict(self.attr_config, save=os.path.join(
                self.attr_dir, 'config.txt'))

    def compute_attributions(self, name, data, target):
        self.approaches[name]['attr'] = []
        start = time()
        for i in range(data.size()[0]):
            if self.verbose:
                print('Method %s | Sample %s / %s' %
                      (name, i+1, data.size()[0]), end='\r')
            self.approaches[name]['attr'].append(
                self.approaches[name]['method'].attribute(
                    data[i].unsqueeze(0), [target[i]]))
        self.approaches[name]['time'] = time() - start
        self.approaches[name]['attr'] = torch.cat(
            self.approaches[name]['attr']).detach().cpu().numpy()

    def get_attribution(self, name, remove_nan=True):
        if type(self.approaches[name]['attr']) is str:
            return np.nan_to_num(np.load(self.approaches[name]['attr'],
                                         allow_pickle=True)[1])
        else:
            return np.nan_to_num(self.approaches[name]['attr'])

    def save_attribution(self, name, path, return_path=False, save_times=True):
        np_file = os.path.join(path, name + '_attr.npy')
        data = np.array(
            [self.attr_config[name],
             self.approaches[name]['attr']], dtype=object)
        np.save(np_file, data)
        if save_times:
            np_file = os.path.join(path, name + '_times.npy')
            data = np.array(
                [self.attr_config[name],
                self.approaches[name]['time']], dtype=object)
        if self.verbose:
            print('Saved file:', np_file)
        if return_path:
            return np_file

    def load_attribution(self, name):
        np_file = os.path.join(self.attr_dir, name + '_attr.npy')
        config, attr = np.load(np_file, allow_pickle=True)
        self.attr_config[name] = config
        self.init_approach(name)
        if not self.save_memory:
            self.approaches[name]['attr'] = attr
        else:
            self.approaches[name]['attr'] = np_file
        if self.verbose:
            print('Loaded file:', np_file)

    def plot_approaches(self, data, index=0, not_show=False, save_path=None):
        approaches_exec = [a for a in sorted(
            self.approaches) if 'attr' in list(self.approaches[a])]
        cols = len(approaches_exec)
        fig, ax = plt.subplots(nrows=3, ncols=cols,
                               figsize=(4*cols, 6), sharey='row')
        fig.suptitle('Attribution methods for sample: ' + str(index))
        axes = ax.flat
        axes[0].set_title('Sample')
        axes[0].plot(data[index].T)
        for i in range(1, cols):
            axes[i].set_visible(False)
        for c, name in enumerate(sorted(approaches_exec)):
            axes[c+cols].set_title(name)
            axes[c+cols].plot(self.get_attribution(name)[index].T)
            axes[c+2*cols].hist(self.get_attribution(name)[index].T,
                                np.arange(0, 1.05, 0.05))

        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        if save_path is not None:
            fname = 'Attribution_Approaches_id-' + str(index) + '.png'
            plt.savefig(os.path.join(save_path, fname), dpi=300,
                        bbox_inches='tight', pad_inches=0.1)

        if not not_show:
            plt.show()
