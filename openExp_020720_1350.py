# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# DICE SM - Exploration & Global Sensitivity Analysis
# Shridhar
# saved 1428

# %% [markdown]
#   ## Setup & Initialisation of levers and uncertainties

# %%
import time

import matplotlib.pyplot as plt
from ema_workbench import (Model, RealParameter, IntegerParameter, ArrayOutcome, 
                           ema_logging, SequentialEvaluator,
                           MultiprocessingEvaluator)
ema_logging.log_to_stderr(ema_logging.INFO)
# from ema_workbench.analysis import pairs_plotting
# from PyDICE_V4_array_outcome import PyDICE
from dicemodel.MyDICE_v3 import PyDICE


# %%

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    

    model = PyDICE()
    dice_sm = Model('dicesmEMA', function=model)
    
    dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                             IntegerParameter('t2xco2_dist',0,2),

                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]
    
    dice_sm.levers = [RealParameter('sr', 0.1, 0.5),
                      RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  0.5, 1.5),
                      RealParameter('emdd',  0.5, 1.5),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]
    
    dice_sm.outcomes = [ArrayOutcome('Atmospheric Temperature'),
                        ArrayOutcome('Damages'),
                        ArrayOutcome('Utility of Consumption'),
                        ArrayOutcome('Savings rate'),
                        ArrayOutcome('Disutility of Damage'),
                        ArrayOutcome('Damage to output ratio'),
                        ArrayOutcome('Welfare'),
                        ArrayOutcome('Total Output')]
    
    
    n_scenarios = 100
    n_policies = 10


# %%
start = time.time()
with MultiprocessingEvaluator(dice_sm, n_processes=8) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(scenarios=n_scenarios,policies=n_policies)
end = time.time()

print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')


# %%
from ema_workbench.analysis import pairs_plotting
fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='policy',legend=True)
fig.set_size_inches(20,20)
plt.show()


# %%
print(outcomes)


# %%
from ema_workbench.analysis import prim
from ema_workbench.analysis import scenario_discovery_util as sdutil
from sklearn import preprocessing 
import numpy as np
x = experiments
# outcome is nd-array, PRIM needs 1D

# y = outcomes['Atmospheric Temperature']
y = np.amax(outcomes['Atmospheric Temperature'], axis =1) 
y


# %%
# len(y)
y.shape


# %%
x.shape


# %%
x


# %%
results_prim = experiments, outcomes
results_prim

# classify_prim = 'Atmospheric Temperature'
# classify_prim = y
classify_prim = np.amax(outcomes['Atmospheric Temperature'], axis =1) 
# classify_prim = outcomes['Atmospheric Temperature']
classify_prim


# %%
prim_setup = prim.setup_prim(results = results_prim, classify = classify_prim, threshold = 0.8)
# prim_setup = prim.setup_prim(x, classify = classify_prim, threshold = 0.8,**kwargs)


# %%
prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha = 0.1)
box1 = prim_alg.find_box()


# %%
box1.show_tradeoff()
plt.show()


# %%


