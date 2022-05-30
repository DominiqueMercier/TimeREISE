import torch


########################################
##### Fast and simple approaches #######
########################################
class FadeMovingAverage:
    def __init__(self):
        pass

    def apply(self, x, masks):
        moving_average = torch.mean(x, 1).view(1, -1, 1)
        masked_x = masks * x.unsqueeze(0) + (1 - masks) * moving_average
        return masked_x


class FadeReference:
    def __init__(self, x_ref=0):
        self.x_ref = x_ref

    def apply(self, x, masks):
        masked_x = self.x_ref + masks * (x - self.x_ref)
        return masked_x


########################################
##### Complex and slower approaches ####
########################################
class FadeMovingAverageWindow:
    def __init__(self, window_size=2, only_past=False):
        self.window_size = window_size
        self.only_past = only_past

    def apply(self, x, masks):
        t_axis = torch.arange(1, x.size()[1] + 1, dtype=int)
        t1_tensor, t2_tensor = t_axis.unsqueeze(1), t_axis.unsqueeze(0)
        t1_t2_sub = t1_tensor - t2_tensor
        if not self.only_past:
            t1_t2_sub = torch.abs(t1_t2_sub)
        filter_coefs = t1_t2_sub <= self.window_size
        filter_coefs = filter_coefs / (2 * self.window_size + 1)
        x_avg = torch.einsum('st,cs->ct', filter_coefs, x).unsqueeze(0)
        masked_x = x_avg + masks * (x - x_avg)
        return masked_x


class GaussianBlur:
    def __init__(self, eps=1.0e-7, sigma_max=2):
        self.sigma_max = sigma_max
        self.eps = eps

    def apply(self, x, masks):
        t_axis = torch.arange(1, x.size()[-1] + 1, dtype=int)
        t1_tensor = t_axis.unsqueeze(1).unsqueeze(2)
        t2_tensor = t_axis.unsqueeze(0).unsqueeze(2)
        # Smooth the mask tensor by applying the kernel
        sigma_tensor = (1 + self.eps) - masks.transpose(1, 2)
        sigma_tensor = (self.sigma_max * sigma_tensor).unsqueeze(1)
        kernel_tensor = torch.exp(-1.0 * (t1_tensor - t2_tensor).unsqueeze(0)
                                  ** 2 / (2.0 * sigma_tensor ** 2))
        kernel_tensor = torch.divide(
            kernel_tensor, torch.sum(kernel_tensor, 1).unsqueeze(1))
        masked_x = torch.einsum('mstc,cs->mct', kernel_tensor, x)
        return masked_x


pert_funcs = {'FadeMovingAverage': FadeMovingAverage,
              'FadeReference': FadeReference,
              'GaussianBlur': GaussianBlur,
              'FadeMovingAverageWindow': FadeMovingAverageWindow}
