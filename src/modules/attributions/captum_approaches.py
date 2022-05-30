import captum.attr as cp
import torch


def normalize(tensor):
    n_samples = tensor.size()[0]
    mi, _ = torch.min(tensor.view(n_samples, -1), dim=1)
    tensor -= mi.view(n_samples, 1, 1)
    ma, _ = torch.max(tensor.view(n_samples, -1), dim=1)
    tensor /= ma.view(n_samples, 1, 1)
    return tensor


########################################
##### Gradient-based approaches ########
########################################
class GuidedBackprop:
    def __init__(self, model):
        self.model = model

    def attribute(self, x, y, norm=True):
        exp = cp.GuidedBackprop(self.model)
        attr = exp.attribute(x, target=y)
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class IntegratedGradients:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.IntegratedGradients(forward_func=self.forward_func)
        base = torch.mean(x, dim=-1, keepdim=True) * \
            torch.ones(x.size()).to(x.device)  # base of the input
        attr = exp.attribute(x, target=y, baselines=base)
        if norm:
            # absolute corresponds to importance
            return normalize(torch.abs(attr))
        return attr


########################################
### Perturbation-based approaches ######
########################################
class FeatureAblation:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.FeatureAblation(self.forward_func)
        attr = exp.attribute(x, target=y)
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class Occlusion:
    def __init__(self, forward_func, window_size=3):
        self.forward_func = forward_func
        self.window_size = window_size

    def attribute(self, x, y, norm=True):
        exp = cp.Occlusion(forward_func=self.forward_func)
        base = torch.mean(x, dim=-1, keepdim=True) * \
            torch.ones(x.size()).to(x.device)  # base of the input
        attr = exp.attribute(x, target=y,
                             sliding_window_shapes=(1, self.window_size),
                             baselines=base)
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


########################################
########### Other approaches ###########
########################################
class Lime:
    def __init__(self, forward_func, n_samples=1000):
        self.forward_func = forward_func
        self.n_samples = n_samples

    def attribute(self, x, y, norm=True):
        exp = cp.Lime(self.forward_func)
        attr = exp.attribute(x, target=y,
                             n_samples=self.n_samples).contiguous()
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr
