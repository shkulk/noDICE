# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


#%%
#  What values of emdd, emuc, prtp_con and prtp_dam correspond to a good, desirable  outcome? What is the probability and confidence level for this?
# What values correspond to particularly bad outcomes? Again, how probable is this, and confident can we be about it?

# How much better is it to use the recommended values for emdd, emuc, prtps, compared to using bad values, or not taking this into consideration? 
# i.e. do we have fewer vulnerable scenarios? How much fewer? Do they form an identifiable subset of the scenarios that the bad policy is vulnerable to?
# 
# %%
from IPython import get_ipython

# DICE SM - PRIM (recovered --> master)

# %% [markdown]
#   ## Setup & Initialisation of levers and uncertainties

# %%

import time
import copy
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
import statsmodels.api as sm
from sklearn import preprocessing 
import ema_workbench.em_framework.evaluators


import os
from dest_directories import gz_path, fig_path
from dicemodel.specs import change_fontsize
from dicemodel.noDICE_v7 import PyDICE
model_version = 'v7'



# %%
from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, TimeSeriesOutcome, ScalarOutcome, ArrayOutcome, Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)
from ema_workbench import save_results, load_results

from ema_workbench.analysis import prim, cart
from ema_workbench.analysis import scenario_discovery_util as sdutil
from ema_workbench.analysis import pairs_plotting, plotting, plotting_util

import ema_workbench.em_framework.evaluators
from dicemodel.specs import change_fontsize, nordhaus_policy
# from ema_workbench.analysis import feature_scoring
# from ema_workbench.em_framework.salib_samplers import get_SALib_problem
# from SALib.analyze import sobol

ema_logging.log_to_stderr(ema_logging.INFO)

# %%

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    

    model = PyDICE()
    dice_sm = Model('dicesmEMA', function=model)
    
    dice_sm.uncertainties = [
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600),
                             RealParameter('emdd', -1, 0.99),
                             IntegerParameter('vd_switch', 0, 1),                             
                            ]
    
    dice_sm.levers = [RealParameter('sr', 0.1, 0.5),
                      RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('emuc',  1.01, 2.00),
                    #   IntegerParameter('vd_switch', 0, 1),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)
                      ]
    
    dice_sm.outcomes = [
                        TimeSeriesOutcome('Atmospheric Temperature'),
                        TimeSeriesOutcome('Total Output'),
                        TimeSeriesOutcome('Per Capita Consumption'),
                        TimeSeriesOutcome('Consumption Growth'),
                        TimeSeriesOutcome('Utility of Consumption'),
                        TimeSeriesOutcome('Per Capita Damage'),
                        TimeSeriesOutcome('Damage Growth'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        TimeSeriesOutcome('Welfare'),
                        # TimeSeriesOutcome('Undiscounted Period Welfare'),
                        TimeSeriesOutcome('Consumption SDR'),
                        TimeSeriesOutcome('Damage SDR'),
                        TimeSeriesOutcome('SCC')
                        ]


# %%
n_scenarios = 100000
n_policy = 1
run = 'run_34_NordOE'
# n_scenarios = 2000
# n_policies = 50
# run = '36_OE'


# %%
## Load results
results = load_results(os.path.join(gz_path,'run_35_NordOE_v7_100000s_.tar.gz'))
experiments, outcomes = results


# %%
# Clean experiments 
# (keep only levers, remove policy, scenario, model columns)
cleaned_experiments = experiments.drop(labels=[l.name for l in dice_sm.uncertainties], axis=1)
cleaned_experiments = experiments.drop(labels= ['policy', 'model',] , axis=1)
x = cleaned_experiments
#  For y: outcomes is nd-array, PRIM needs 1D
#%%
# Clean outcomes
# dropping first two steps (warm up) and last five steps(cooldown)
cleaned_outcome = {}
for key, value in outcomes.items():
    cleaned_outcome[key] = value[:,2:-5]  
cleaned_outcome['Welfare'].shape
# %%
# values for 2300 
end_outcome = {}
for key, value in outcomes.items():
    end_outcome[key] = value[:, -1]  

# np.mean(end_outcome['Damage Growth'], axis =0)

#%%
y = copy.deepcopy(cleaned_outcome)
y_end = copy.deepcopy(end_outcome)

# %%
# experiments_np = experiments.to_records()
# experiments_np


# %%
## reduce by Binary classification of results
# what range of outcome vars represents bad outcome - we want to find the 

######## by value

# Atmospheric Temperature
# what causes the highest values of this outcome (> 2 degrees)
y_temp = np.amax(y['Atmospheric Temperature'], axis =1)  > 2.0


# Utility of Consumption
# what causes the values  of U(C) higher than 0.5 (from the violin plot)
y_utility_con = end_outcome['Utility of Consumption'] > 0.5
#%%
# Disutility of Damage

data_disutil_dam = np.max(cleaned_outcome['Disutility of Damage'], axis = 1)
# y_disutil_dam_top20 = data_disutil_dam > np.percentile(data_disutil_dam, 80)
y_disutil_dam_gr3 = data_disutil_dam > 3
# y_disutil_dam10 = end_outcome['Disutility of Damage'] > np.percentile(end_outcome['Disutility of Damage'], 10)

###### by percentile
# #percentile < 80 = bottom 80
#%%
# Welfare
# y_welfare = end_outcome['Welfare'] < 5000
y_welfare10 = end_outcome['Welfare'] < np.percentile(end_outcome['Welfare'], 10)

#%%

# Per capita Damage
data_damage = np.amax(y['Per Capita Damage'], axis=1)
y_dpc5 = data_damage < np.percentile(data_damage, 5)
#%%
# what causes the low number of high values of damage outcomes (thin top tail): values below top 80th percentile(i.e. apart from the most common results seen)
y_dpc = end_outcome['Per Capita Damage'] < np.percentile(end_outcome['Per Capita Damage'], 10)
#%%
# Consumption Per Capita 
# what causes the low number of low values per capita consumption (thin bottom tail): bottom 20th percentile
y_cpc = end_outcome['Per Capita Consumption'] < np.percentile(end_outcome['Per Capita Consumption'], 20)


data_con_g = np.amax(y['Consumption Growth'], axis =1)
y_con_g = data_con_g < 0.0
# y_con_g = data_con_g < np.percentile(data_con_g, 20) # what causes the least (bottom 20th percentile) of this outcome
# np.any(data_con_g)
#%%
# data_output = y_end['Total Output']
data_output = np.amax(y['Total Output'], axis=1)
#%%
y_output_bottom20 = data_output < np.percentile(data_output, 20) # what causes the least values (bottom 20th percentile) of this outcome

# %%

x = cleaned_experiments
y = y_disutil_dam_gr3

prim_alg = prim.Prim(x, y, threshold=0.5, peel_alpha=0.1) #0.1 

# the meaning of peel_alpha is the percentile of the data that is to be removed
# The peeling alpha determines how much data is peeled off in each iteration of the algorithm. The lower the value, the less data is removed in each iteration. Controls the leniency of the algorithm, the higher the less lenient.
# from ema_workbench.analysis import prim


# %%
box1 = prim_alg.find_box()
box1.peeling_trajectory
#%%
box1.show_tradeoff()
# %%
box1.inspect(19)
#%%
box1.inspect(22, style='graph')
plt.show()

#%%
box1.select(2)
box1.show_pairs_scatter()
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.show()
fig.savefig(os.path.join(fig_path,str(run) + '_PRIM_welfare_box1' + '.png'))

# %%
box2 = prim_alg.find_box()
box2.peeling_trajectory
box2.inspect()
box2.show_tradeoff()
box2.inspect(49, style='graph')

#%%
box2.show_pairs_scatter()
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.show()
fig.savefig(os.path.join(fig_path,str(run) + '_PRIM_Uwel_box2' + '.png'))


# %%
box3 = prim_alg.find_box()
box3.inspect()
box3.show_tradeoff()
box3.inspect(14, style='graph')


# %%
print (prim_alg.stats_to_dataframe())


# %%
box1.select(12)
box1.show_pairs_scatter()
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.show()


# %%
# boxes = prim_alg.show_boxes()
prim.Prim.show_boxes(prim_alg)
# #visualize
# prim.show_boxes_individually(boxes, results)
# prim.show_boxes_together(boxes, results)
# plt.show()


# %%
# CART
from ema_workbench.analysis import cart
cart_alg = cart.CART(x,y_welfare10, 0.05)
cart_alg.build_tree()

print (cart_alg.stats_to_dataframe())
print(cart_alg.boxes_to_dataframe())
cart_uWel_df = cart_alg.boxes_to_dataframe()

# %%
# dice_sm.constants
# ema_workbench.analysis.kde_over_time()

# ema_workbench.analysis.lines()

# ema_workbench.analysis.plotting_util

# perform_experiments()