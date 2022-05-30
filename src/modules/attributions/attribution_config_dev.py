import numpy as np

########################################
######### Attribution Config ###########
########################################
config = {}

# working
config['GuidedBackprop'] = {'execute': True}

config['IntegratedGradients'] = {'execute': True}

config['FeatureAblation'] = {'execute': True}

config['Occlusion'] = {'execute': True}
config['Occlusion']['config'] = {'window_size': 3}

config['Lime'] = {'execute': True}
config['Lime']['config'] = {'n_samples': 10000}


config['Timereise'] = {'execute': True}
config['Timereise']['config'] = {'perturbation': 'FadeReference',
                                 'n_masks': 1000,
                                 'probability': np.linspace(0.05, 0.95, 10),
                                 'granularity': np.linspace(0.05, 0.05, 1),
                                 'relative': False,
                                 'load': False}
config['Timereise']['pert_config'] = {}
