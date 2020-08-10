# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%%
import os

myfolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(myfolder)

from IPython import get_ipython
import time
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
import statsmodels.api as sm
from sklearn import preprocessing 
import ema_workbench.em_framework.evaluators

from dicemodel.noDICE_v6 import PyDICE
# from noDICE_v5 import PyDICE
model_version = 'v6'


from dest_directories import gz_path, fig_path


# %%
from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, TimeSeriesOutcome, ScalarOutcome, ArrayOutcome, Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)
from ema_workbench import save_results, load_results
# from ema_workbench.analysis import prim, cart
from ema_workbench.analysis import scenario_discovery_util as sdutil
from ema_workbench.analysis import pairs_plotting, plotting, plotting_util


ema_logging.log_to_stderr(ema_logging.INFO)


# %%

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    

    model = PyDICE()
    dice_sm = Model('dicesmEMA', function=model)
    
    dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                            #  IntegerParameter('t2xco2_dist',0,2),
                            #  IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600),
                             RealParameter('emdd', -1.0, 0.99),
                            ]
    
    dice_sm.levers = [RealParameter('sr', 0.1, 0.5),
                      RealParameter('prtp_con',  0.001, 0.015),
                    #   RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  1.01, 2.00),IntegerParameter('VD_switch', 0, 1),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)
                      ]
    
    dice_sm.outcomes = [
                        TimeSeriesOutcome('Atmospheric Temperature'),
                        TimeSeriesOutcome('Total Output'),
                        # TimeSeriesOutcome('Per Capita Consumption'),
                        # TimeSeriesOutcome('Consumption Growth'),
                        TimeSeriesOutcome('Utility of Consumption'),
                        # TimeSeriesOutcome('Per Capita Damage'),
                        # TimeSeriesOutcome('Damage Growth'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        TimeSeriesOutcome('Welfare'),
                        # TimeSeriesOutcome('Undiscounted Period Welfare'),
                        TimeSeriesOutcome('Consumption SDR'),
                        TimeSeriesOutcome('Damage SDR')
                        ]


# %%
n_scenarios = 2000
n_policies = 20

run = 8

# %% Sequential processing
start = time.time()
with SequentialEvaluator(dice_sm) as evaluator:
    results = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)
end = time.time()

print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')


# %% Multiprocessing
start = time.time()
with MultiprocessingEvaluator(dice_sm, n_processes=8) as evaluator:
    results = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)
end = time.time()

print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')

# %%
start = time.time()
save_results(results, os.path.join(gz_path, str(run) + '_OE_' + str(n_scenarios) + 's_' + str(n_policies) + 'p_' + '.tar.gz'))
end = time.time()
print('Saving time is ' + str(round((end - start)/60)) + ' mintues')

# %%
results = load_results(os.path.join(gz_path,'020820_OE_1000s_20p_.tar.gz'))
# outcomes
#%%
experiments, outcomes = results

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
    end_outcome[key] = value[:,-1]  

# np.mean(end_outcome['Damage Growth'], axis =0)

#%%
########## Prepare outcomes as needed

# remove outcomes that you don't need
rem_list = [
    # 'Atmospheric Temperature',
    # 'Per Capita Consumption',
    # 'Consumption Growth',
    # 'Utility of Consumption',
    # 'Per Capita Damage',
    # 'Damage Growth',
    # 'Disutility of Damage',
    # 'Welfare',
    # 'Undiscounted Period Welfare',
    # 'Consumption SDR',
    # 'Damage SDR'
]
for key in rem_list:
    end_outcome.pop(key) 


# %%
############## Pairs scatter plot: grouped by policy

fig, axes = ema_workbench.analysis.pairs_plotting.pairs_scatter(experiments, outcomes,group_by='policy')
fig.set_size_inches(20,20)
plt.show()

# %%
fig.savefig(os.path.join(fig_path,'OE_pairsplot_by_policy_' + str(n_scenarios) + 's' + str(n_policies) + 'p' + '.png'))

# %% 
## Pairs plotting by emdd range intervals using ema workbench 

grouping_specifiers = {'a':1, 'b':2, 'c':3}
grouping_labels = sorted(grouping_specifiers.keys())
grouping_specifiers = [grouping_specifiers[key] for key in
                                       grouping_labels]
low = np.min(experiments['emdd'])
high = np.max(experiments['emdd'])
grouping_specifiers_emdd = {'Low': low, 'mid': 0, 'High': high}

# %%

fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='emdd',grouping_specifiers=grouping_specifiers_emdd, legend=True, transparent=True, papertype='letter')
fig.set_size_inches(20,20)
plt.show()

# %%
repeat_token = 1

fig.savefig(os.path.join(fig_path,'OE_pairsplot_by_emdd_ema' + str(n_scenarios) + 's' + str(n_policies) + 'p_' + str(repeat_token) + '.png'))


#%%
# Pairs plotting grouped by emdd intervals through sns

out_DF = pd.DataFrame(end_outcome)
out_DF.head()
emdd = experiments['emdd']
out_DF['emdd'] = emdd

# low = -0.5
# %%
# RealParameter('emdd', -1.0, 0.99)
# out_DF['emdd'] = out_DF['emdd'].apply(lambda x: '-0.99 to -0.5' if x < -0.5 else('-0.5 to 0' if (x < 0) else('0 to 0.5' if (x < 0.5) else '0.5 to 0.99')))
#%%
out_DF['emdd'] = out_DF['emdd'].apply(
    lambda x: '-0.99 to -0.5' if x < -0.5
        else('-0.5 to 0' if (x < 0)
            else('0 to 0.5' if (x < 0.5)
                else ('0.5 to 0.99' if (x < 0.99) 
                    else('0.99 to 1.45' if (x < 1.45)
                        else '1.45 to 2'
                        )
                    )
                )
            )
    )


# %%
# out_DF = out_DF.drop(columns=['Total Output', 'Per Capita Consumption', 'Per Capita Damage','Damage SDR' ])

# %%
# clr_palette = ([sns.cubehelix_palette(8)[6],sns.color_palette("inferno", 15)[-2],sns.color_palette("YlGn", 15)[10]])

# clr_palette = zip(df['category'].unique(), sns.crayons.values())
clr_palette = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

# %%
sns.set_style("whitegrid")
sns_plot = sns.pairplot(out_DF, hue='emdd', palette=clr_palette, vars=list(end_outcome.keys()), height=2)
# fig.set_size_inches(20, 20)
sns_plot.savefig(os.path.join(fig_path,'20k_pairplot_2300_by_emdd_range' +'.png'))
plt.show()

# %%
# Time Series Plotting

TimeLine = []
for i in range(65):
    TimeLine.append(2010+i*5)
TimeLine
outcomes["TIME"] = np.array([TimeLine])
outcomes

# %%
plt.close('all')
#%%
#  Time series outcome

for outcome in outcomes.keys():
    fig,axes=plotting.lines(experiments, outcomes, outcomes_to_show=outcome, density=plotting.Density.BOXENPLOT, legend=True)
    
    fig.set_size_inches(25, 10)
    
    repeat_token=2
    fig.savefig(os.path.join(fig_path, str(repeat_token) + 'OE_TimeSeries' + str(outcome) +  '_.png'))
    
plt.show()

from matplotlib import axes
# %%
# Time series outcome grouped by policy

for outcome in outcomes.keys():
    fig,axes=plotting.lines(experiments, outcomes, outcomes_to_show=outcome, density=plotting.Density.BOXENPLOT, group_by='policy', legend=True)
    # fig.tight_layout()

    fig.set_size_inches(15, 10)
    
    repeat_token = 1
    fig.savefig(os.path.join(fig_path,'1a_Time_grouped_' + str(outcome) + str(repeat_token) + '.png'))
plt.show()


#%%
# Bramka
welfare_last = outcomes['Welfare'][:, -1]
# : is copy all
outcomes['Welfare'].shape  # (20000, 65) i.e. (experiments, tsteps)
# I want to remove elements only along the tsteps axis, not the experiment axis. 

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11,12,13,14,15]])
arr.shape
arr[:, 1:-2]

welfare_last = outcomes['Welfare'][:,2:-5 ]

