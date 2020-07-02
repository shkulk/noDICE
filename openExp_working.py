
import os
import sys
# os.chdir(os.getcwd())
# pydice_folder = os.path.dirname(os.getcwd()) + "\\06_Code"
# sys.path.insert(1, pydice_folder)
# pydice_folder
# %%
# DICE SM - Exploration & Global Sensitivity Analysis
# Shridhar

# %% [markdown]
#  ## Setup & Initialisation of levers and uncertainties

# %%
import time

import matplotlib.pyplot as plt
from ema_workbench import (Model, RealParameter, IntegerParameter, ArrayOutcome, 
                           ema_logging, SequentialEvaluator,
                           MultiprocessingEvaluator)
ema_logging.log_to_stderr(ema_logging.INFO)

# from PyDICE_V4_array_outcome import PyDICE
from dicemodel.MyDICE_v3 import PyDICE


# %%

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    

    model = PyDICE()
    dice_sm = Model('dicesmEMA', function=model)
    
    dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                             IntegerParameter('t2xco2_dist',0,2),
                            #  IntegerParameter('fdamage',0,2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]
    
    dice_sm.levers = [RealParameter('sr', 0.1, 0.5),
                      #RealParameter('prtp_con',  0.001, 0.015),
                      #RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  0.5, 1.5),
                      RealParameter('emdd',  0.5, 1.5),
                      #IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]
    
    dice_sm.outcomes = [ArrayOutcome('Atmospheric Temperature'),
                        ArrayOutcome('Damages'),
                        ArrayOutcome('Utility of Consumption'),
                        ArrayOutcome('Savings rate'),
                        ArrayOutcome('Disutility of Damage'),
                        ArrayOutcome('Damage to output ratio'),
                        ArrayOutcome('Welfare'),
                        ArrayOutcome('Total Output')]
    
    
    n_scenarios = 10
    n_policies = 10
    
    start = time.time()
    with MultiprocessingEvaluator(dice_sm) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=n_scenarios,policies=n_policies)
    end = time.time()
    
    print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')



#Importing the needed Ema workbench libraries

# from ema_workbench import (Model, Policy, CategoricalParameter,ScalarOutcome, IntegerParameter, RealParameter, optimize, Scenario)

# from ema_workbench import ema_logging, MultiprocessingEvaluator, SequentialEvaluator

# from ema_workbench.em_framework.optimization import EpsilonProgress, HyperVolume

# from ema_workbench import (MultiprocessingEvaluator, ema_logging, perform_experiments, Constant)

# from ema_workbench.em_framework.salib_samplers import get_SALib_problem

# from ema_workbench.em_framework.samplers import sample_levers, sample_uncertainties

# from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS, SequentialEvaluator, BaseEvaluator


# fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='policy',legend=True)

# %%
