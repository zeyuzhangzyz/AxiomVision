import json
import numpy as np
S = 1
beta = 0.25
beta_p = 1.5
phase_cardinality = 2
# you can change the visual_models_begin and visual_models_end to change the AxiomVision levels divide.
# when will you choose different level depends on the is_final and last_element.
visual_models_begin = [0, 11, 15]
visual_models_end = [11, 15, 17]
prob = np.random.uniform(0,1)
real = 1
change_real = 1
real_payoff = None
svd_payoff = None
svd_result = None
best_fix = 0
acc_threshold = 0.5
reconnect = False
def load_config(config_path='Algorithm/experiment_config.json'):
    global S, beta, visual_models_begin, visual_models_end, prob, real, change_real,best_fix, acc_threshold
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        S = config['S']
        beta = config['beta']
        visual_models_begin = config['visual_models_begin']
        visual_models_end = config['visual_models_end']
        prob = np.random.uniform(0,1)
        real = config['real']
        change_real = config['change_real']
        best_fix = config['best_fix']
        acc_threshold = config['acc_threshold']

