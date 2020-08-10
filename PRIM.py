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
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
# import statsmodels.api as sm

from sklearn import preprocessing 


# %%
from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, TimeSeriesOutcome, ScalarOutcome, ArrayOutcome, Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)
from ema_workbench import save_results, load_results

from ema_workbench.analysis import prim, cart
from ema_workbench.analysis import scenario_discovery_util as sdutil
from ema_workbench.analysis import pairs_plotting, plotting, plotting_util

import ema_workbench.em_framework.evaluators
# from ema_workbench.analysis import feature_scoring
# from ema_workbench.em_framework.salib_samplers import get_SALib_problem
# from SALib.analyze import sobol

ema_logging.log_to_stderr(ema_logging.INFO)


# %%
from dicemodel.noDICE_v4 import PyDICE
model_version = '4'

import os
from dest_directories import gz_path, fig_path


# %%

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    

    model = PyDICE()
    dice_sm = Model('dicesmEMA', function=model)
    
    dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                             IntegerParameter('t2xco2_dist',0,2),
                             IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                            #  IntegerParameter('cback', 100, 600)
                            ]
    
    dice_sm.levers = [RealParameter('sr', 0.2, 0.3),
                      RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  1.01, 2.00),
                      RealParameter('emdd', -1.0, 0.99),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)
                      ]
    
    dice_sm.outcomes = [TimeSeriesOutcome('Atmospheric Temperature'),
                        TimeSeriesOutcome('Total Output'),
                        # TimeSeriesOutcome('Population'),
                        TimeSeriesOutcome('Per Capita Consumption'),
                        TimeSeriesOutcome('Consumption Growth'),
                        TimeSeriesOutcome('Utility of Consumption'),
                        TimeSeriesOutcome('Per Capita Damage'),
                        TimeSeriesOutcome('Damage Growth'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        TimeSeriesOutcome('Welfare'),
                        TimeSeriesOutcome('Undiscounted Period Welfare'),
                        TimeSeriesOutcome('Consumption SDR'),
                        TimeSeriesOutcome('Damage SDR')

                        ]


# %%
n_scenarios = 100
n_policies = 50


# %%
## Load results
results = load_results(os.path.join(gz_path,'v4_fdamage_1a_OE_sr100s_20p_.tar.gz'))
experiments, outcomes = results


# %%
# for x: Clean experiments (keep only levers, remove policy, scenario, model columns)
cleaned_experiments = experiments.drop(labels=[l.name for l in dice_sm.uncertainties], axis=1)
cleaned_experiments = experiments.drop(labels= ['policy', 'scenario', 'model',] , axis=1)
# type(cleaned_experiments)
# cleaned_experiments
x = cleaned_experiments
#  For y: outcomes is nd-array, PRIM needs 1D
#%%

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
y = cleaned_outcome


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

# Undiscounted period welfare : from graph, and equal to cases when U(C) > V(D)
y_undiscounted_period_welfare = end_outcome['Undiscounted Period Welfare'] < 0.0

# Utility of Consumption
# what causes the values  of U(C) higher than 0.5 (from the violin plot)
y_utility_con = end_outcome['Utility of Consumption'] > 0.5

# Disutility of Damage
# what causes the values  of V(D) higher than 5 (from the violin plot, U(C) range never exceeds 4)
y_disutil_dam = end_outcome['Disutility of Damage'] > 4


###### by percentile
# #percentile < 80 = bottom 80

# Welfare

y_welfare = end_outcome['Welfare'] < np.percentile(end_outcome['Welfare'], 90)



# Per capita Damage
# what causes the low number of high values of damage outcomes (thin top tail): values below top 80th percentile(i.e. apart from the most common results seen)
y_dpc = end_outcome['Per Capita Damage'] < np.percentile(end_outcome['Per Capita Damage'], 80)

# Consumption Per Capita 
# what causes the low number of low values per capita consumption (thin bottom tail): bottom 20th percentile
y_cpc = end_outcome['Per Capita Consumption'] < np.percentile(end_outcome['Per Capita Consumption'], 20)


data_con_g = np.amax(y['Consumption Growth'], axis =1)
y_con_g = data_con_g < 0.0
# y_con_g = data_con_g < np.percentile(data_con_g, 20) # what causes the least (bottom 20th percentile) of this outcome
# np.any(data_con_g)

data_dam_g = np.amax(y['Damage Growth'], axis =1)
y_dam_g = data_dam_g < np.percentile(data_dam_g, 20)



data_output = np.amax(y['Total Output'], axis = 1)
y_output = data_output < np.percentile(data_output, 20) # what causes the least values (bottom 20th percentile) of this outcome

# %%

# the meaning of peel_alpha is the percentile of the data that is to be removed
# The peeling alpha determines how much data is peeled off in each iteration of the algorithm. The lower the value, the less data is removed in each iteration. Controls the leniency of the algorithm, the higher the less lenient.
# from ema_workbench.analysis import prim
x = cleaned_experiments
y = y_welfare

prim_alg = prim.Prim(x, y, threshold=0.5, peel_alpha=0.1) #0.1 


# %%
box1 = prim_alg.find_box()
box1.peeling_trajectory

# %%
box1.show_tradeoff()
box1.inspect(13)
box1.inspect(0, style='graph')

plt.show()

#%%
box1.select(21)
box1.show_pairs_scatter()
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.show()
fig.savefig(os.path.join(fig_path,str(run) + '_PRIM_Uwel_box1' + '.png'))

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
box1.select(21)
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
cart_alg = cart.CART(x,y_welfare, 0.05)
cart_alg.build_tree()

print (cart_alg.stats_to_dataframe())
print(cart_alg.boxes_to_dataframe())
cart_uWel_df = cart_alg.boxes_to_dataframe()

# %%
