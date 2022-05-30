import os

import numpy as np
import torch
from modules.attributions.timereise.perturbations import pert_funcs
from scipy.ndimage import uniform_filter


class Timereise():
    def __init__(self, model_forward, input_size, n_masks=10000,
                 probability=0.1, granularity=0.05,
                 perturbation='FadeReference', n_batch=32, relative=False,
                 mask_path=None, load=True, random_state=0, device='cuda'):
        self.model_forward = model_forward
        self.input_size = input_size
        self.n_masks = n_masks
        self.probability = [probability] if type(
            probability) is float else probability
        self.granularity = [granularity] if type(
            granularity) is float else granularity
        if type(perturbation) is str:
            self.perturbation = perturbation
        else:
            self.perturbation = pert_funcs[perturbation]()
        self.n_batch = n_batch
        self.relative = relative
        self.mask_path = mask_path
        self.load = load
        self.random_state = random_state
        self.device = device

        # compute actual mask distribution
        self.retrieve_actual_n_masks()
        # load masks
        if self.load and self.mask_path is not None and \
                os.path.exists(self.mask_path):
            self.load_masks(self.mask_path)
        else:
            # directly generate masks and save if path set
            self.generate_masks(self.mask_path)

    ########################################
    ############### Mask setup #############
    ########################################
    def retrieve_actual_n_masks(self):
        if self.relative:
            expectation = np.matmul(np.expand_dims(self.probability, 1),
                                    np.expand_dims(self.granularity, 0))
            rel_exp = np.max(expectation) / expectation
            norm_exp = rel_exp / np.sum(rel_exp)
            self.n_masks_setup = np.ceil(self.n_masks * norm_exp).astype(int)
            self.n_masks_all = np.sum(self.n_masks_setup)
        else:
            # use same number of masks for all setups
            self.n_masks_setup = np.ones((len(self.probability),
                                          len(self.granularity)),
                                         dtype=int) * self.n_masks
            self.n_masks_all = len(self.probability) * \
                len(self.granularity) * self.n_masks

    def generate_masks(self, mask_path=None):
        masks = []
        rng = np.random.RandomState(self.random_state)
        for i, prob in enumerate(self.probability):
            for j, grain in enumerate(self.granularity):
                tmp_masks = self.create_masks(rng, prob, grain,
                                              self.n_masks_setup[i, j])
                masks.append(tmp_masks)
        self.masks = torch.cat(masks, dim=0)  # (N_masks, H, W)
        if mask_path is not None:
            self.save_masks(mask_path)

    def create_masks(self, rng, prob, grain, n_masks):
        masks = []
        # cell size in the upsampled mask
        n_slices = int(np.ceil(1 / grain))
        slice_width = np.ceil(self.input_size[1] / n_slices)
        resize_w = int((n_slices + 1) * slice_width)
        # loop to provide required amount of random masks
        for _ in range(n_masks):
            # generate binary mask
            binary_mask = torch.Tensor(rng.rand(
                1, self.input_size[0], n_slices))
            binary_mask = (binary_mask < prob).float()
            # upsampling mask
            mask = torch.nn.functional.interpolate(
                binary_mask, resize_w, mode='linear',
                align_corners=False)
            # random cropping
            j = rng.randint(0, slice_width)
            mask = mask[:, :, j:j+self.input_size[1]]
            masks.append(mask)  # append to complete mask list
        masks = torch.cat(masks, dim=0)  # (N_masks, H, W)
        return masks

    ########################################
    ######### Persistent Masks #############
    ########################################
    def save_masks(self, path):
        torch.save(self.masks, path)

    def load_masks(self, path):
        self.masks = torch.load(path)

    ########################################
    ########## Mask processing #############
    ########################################
    def forward(self, x, mask_func=None, activation_func=torch.softmax,
                smooth=0, mask_ids=None, return_outs=False,
                relative=None, score_func=None):
        # use default perturbation if none is given
        if mask_func is None:
            mask_func = self.perturbation
        # subset masking if not all masked should be used
        used_masks = self.masks if mask_ids is None else self.masks[mask_ids]

        outs = []  # keep probabilities of each class
        masked_x = mask_func.apply(x, used_masks)  # (n_masks, C, T)
        for i in range(0, used_masks.size()[0], self.n_batch):
            input_x = masked_x[i:i + self.n_batch]
            out = self.model_forward(input_x.to(self.device))
            outs.append(out.detach().to('cpu'))
        outs = torch.cat(outs)  # (n_masks, n_classes)

        if activation_func is not None:
            outs = activation_func(outs, axis=1)

        score = outs.clone()
        # relative used for regression
        if relative:
            ref = self.model_forward(x.to(self.device))
            if activation_func is not None:
                ref = activation_func(ref, axis=1)
            score = score_func(ref, outs)

        # computation of saliency
        saliency = self.compute_saliency(score, used_masks)
        saliency = self.normalize(saliency)

        # smooth maps
        if smooth > 0:
            saliency = torch.Tensor(uniform_filter(saliency.numpy(),
                                                   size=(1, 1, smooth)))

        if return_outs:
            return saliency, outs
        return saliency

    def compute_saliency(self, probs, masks):
        n_classes = probs.shape[1]
        n_masks_all = masks.size()[0]
        # caluculate saliency map using probability scores as weights
        saliency = torch.matmul(probs.transpose(
            0, 1), masks.view(n_masks_all, -1))
        saliency = saliency.view(
            (n_classes, *self.input_size))
        saliency = torch.divide(saliency, torch.sum(masks, dim=0).unsqueeze(0))
        return saliency

    def normalize(self, saliency):
        n_classes = saliency.size()[0]
        mi, _ = torch.min(saliency.view(n_classes, -1), dim=1)
        saliency -= mi.view(n_classes, 1, 1)
        ma, _ = torch.max(saliency.view(n_classes, -1), dim=1)
        saliency /= ma.view(n_classes, 1, 1)
        return saliency

    # wrapper to pass multiple samples and select targets
    def attribute(self, x, y=None, pert=None):
        attr = []
        for i in range(x.size()[0]):
            sal = self.forward(x[i].to('cpu'), pert)
            if y is not None:
                if type(y) is int or type(y) is np.int64:
                    sal = sal[y]
                else:
                    sal = sal[y[i]]
            attr.append(sal.view(1, *sal.size()))
        attr = torch.cat(attr)
        return attr
