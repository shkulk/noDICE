#%%
import time
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
import statsmodels.api as sm
from sklearn import preprocessing 
import ema_workbench.em_framework.evaluators
import matplotlib.pyplot as plt

import os
from dest_directories import gz_path, fig_path
from dicemodel.noDICE_v6 import PyDICE
model_version = 'v6'

from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, TimeSeriesOutcome, ScalarOutcome, ArrayOutcome, Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)
from ema_workbench import save_results, load_results
from ema_workbench.analysis import prim, cart
from ema_workbench.analysis import scenario_discovery_util as sdutil
from ema_workbench.analysis import plotting, plotting_util

ema_logging.log_to_stderr(ema_logging.INFO)

######



######
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
                             RealParameter('emdd', -0.5, 0.99),
                            ]
    
    dice_sm.levers = [RealParameter('sr', 0.1, 0.5),
                      RealParameter('prtp_con',  0.001, 0.015),
                    #   RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc', 1.01, 2.00),
                      IntegerParameter('vd_switch', 0, 1),
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
run = 9


# %%

start = time.time()
with MultiprocessingEvaluator(dice_sm, n_processes=8) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)
end = time.time()

print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')

# %%
results = experiments, outcomes

start = time.time()
save_results(results, os.path.join(gz_path, str(run) + str(model_version)+'_OE_' + str(n_scenarios) + 's_' + str(n_policies) + 'p_' + '.tar.gz'))
end = time.time()

print('Saving time is ' + str(round((end - start)/60)) + ' mintues')

# %%
results = load_results(os.path.join(gz_path,'9v6_OE_2000s_20p_.tar.gz'))
experiments, outcomes = results

# %%
# experiment debug (number of unique values for each variable)
experiments.info()
unique_counts = pd.DataFrame.from_records([(col, experiments[col].nunique()) for col in experiments.columns],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
unique_counts
# experiments['policy']

# %% [markdown]
# # Prepare outcomes as needed

# %%
#  dropping first two steps warm-up period and last 5 steps cooldown period cooldown periods

# outcomes[key].shape from (20000, 65) to (20000,58)

for key, value in outcomes.items():
    outcomes[key] = value[:,2:-5]  
outcomes['Welfare'].shape


# np.mean(cleaned_outcome['Damage Growth'], axis =0)


# %%
# remove outcomes that you dont need for pairs plots and time series
rem_list = [
    # 'Atmospheric Temperature',
    # 'Total Output',
    # 'Per Capita Consumption',
    # 'Consumption Growth',
    # 'Utility of Consumption',
    # 'Per Capita Damage',
    # 'Damage Growth',
    # 'Disutility of Damage',
    # 'Welfare',
    # 'Undiscounted Period Welfare',
    'Consumption SDR',
    'Damage SDR'
]
for key in rem_list:
    # end_outcome.pop(key)
    # cleaned_outcome.pop(key)
    outcomes.pop(key)

# %%
outcomes.keys()

# %%
# Values for 2300
end_outcome = {}
for key, value in outcomes.items():
    end_outcome[key] = value[:, -1]  
end_outcome['Welfare'].shape

# %% [markdown]
# # Pairs plotting grouped by policy

# %%

from ema_workbench.analysis import pairs_plotting
fig, axes = ema_workbench.analysis.pairs_plotting.pairs_scatter(experiments, outcomes,group_by='policy')
fig.set_size_inches(20,20)

repeat_token = 1
fig.savefig(os.path.join(fig_path, str(run) + '_v6_OE_pairs_grouped_policy_' + str(repeat_token) + '.png'))

plt.show()

# %% [markdown]
# # Pairs plotting using ema workbench grouped by V(D) switch on/ off

# %%
grouping_specifiers_VD = {'without V(D)':0, 'with V(D)':1}
grouping_labels_VD = (grouping_specifiers_VD.keys())
# grouping_labels = sorted(grouping_specifiers.keys())
# grouping_specifiers_VD = [grouping_specifiers[key] for key in grouping_labels]

# %%
fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='vd_switch',grouping_specifiers=grouping_specifiers_VD, legend=True, transparent=True)
fig.set_size_inches(20,20)
plt.show()


# %%
fig.savefig(os.path.join(fig_path, str(run) + '__v6_pairsplot_by_VDswitch_ema' + str(n_scenarios) + 's_' + str(n_policies) + 'p_' +  '.png'))

# %% [markdown]
# # Pairs plotting using ema workbench grouped by emdd range intervals 
# (this isn't really working out)

# %%
grouping_specifiers_emdd = {'-0.1 to -0.5': -0.5, '-0.5 to 0': -0, '0 to 0.5': 0.5,'0.5 to 0.99': 0.99}
grouping_labels = (grouping_specifiers_emdd.keys())
# grouping_specifiers = [grouping_specifiers[key] for key in grouping_labels]
#%%
out_DF = pd.DataFrame(end_outcome)
out_DF.head()
emdd = experiments['emdd']
out_DF['emdd'] = emdd
# %%
from ema_workbench.analysis import pairs_plotting, plotting_util
fig, axes = pairs_plotting.pairs_scatter(experiments,end_outcome, group_by='emdd',grouping_specifiers=grouping_specifiers_emdd, 
legend=True, 
)
fig.set_size_inches(20,20)
plt.show()

# %% [markdown]
# # Pairs plotting by EMDD range through sns

# %%
clr_palette = ([sns.cubehelix_palette(8)[6],sns.color_palette("inferno", 15)[-2],sns.color_palette("YlGn", 15)[10]])


# %%
out_DF = pd.DataFrame(end_outcome)
out_DF.head()
emdd = experiments['emdd']
out_DF['emdd'] = emdd


# %%
out_DF['emdd'] = out_DF['emdd'].apply(
    lambda x: '-1 to -0.5' if x < -0.5
        else('-0.5 to 0' if (x < 0)
            else('0 to 0.5' if (x < 0.5)
                else ('0.5 to 0.99')
                )
            )
    )


# %%
sns_plot = sns.pairplot(out_DF, hue='emdd', vars=list(end_outcome.keys())) # palette=clr_palette
fig.set_size_inches(20, 20)

sns_plot.savefig(os.path.join(fig_path, str(run) + '_v6_snspairplot_by_emdd_2330' +'.png'))
plt.show()


# %% [markdown]
# # Time Series Plotting

# %%
# %%
results = load_results(os.path.join(gz_path,'8_OE_2000s_20p_.tar.gz'))
experiments, outcomes = results

#%%
# Add a 'TIME' outcome 
TimeLine = []
for i in range(65):
    TimeLine.append(2010+i*5)
# TimeLine
outcomes["TIME"] = np.array([TimeLine])
outcomes["TIME"].shape

#%%
#  dropping first two steps warm-up period and last 5 steps cooldown period cooldown periods

# outcomes[key].shape from (20000, 65) to (20000,58)

for key, value in outcomes.items():
    outcomes[key] = value[:,2:-5]  
outcomes['Welfare'].shape


# np.mean(cleaned_outcome['Damage Growth'], axis =0)


# %%
# remove outcomes that you dont need for pairs plots and time series
rem_list = [
    # 'Atmospheric Temperature',
    'Total Output',
    # 'Per Capita Consumption',
    'Consumption Growth',
    # 'Utility of Consumption',
    # 'Per Capita Damage',
    'Damage Growth',
    # 'Disutility of Damage',
    # 'Welfare',
    # 'Undiscounted Period Welfare',
    # 'Consumption SDR',
    # 'Damage SDR'
]
for key in rem_list:
    # end_outcome.pop(key)
    # cleaned_outcome.pop(key)
    outcomes.pop(key)


# %%
experiments.shape


# %%
# sns.set_style("whitegrid")

for outcome in outcomes.keys():
    fig,axes=plotting.lines(experiments, outcomes,outcomes_to_show=outcome, density=plotting.Density.BOXENPLOT, legend=True)
    
    fig.set_size_inches(25,10)
    fig.savefig(os.path.join(fig_path, str(run) + '_v6_OE_TimeSeries_' + str(outcome) + '.png'))
plt.show()


# %%
# Time series outcome grouped by V(D) switch

for outcome in outcomes.keys():
    fig,axes=plotting.lines(experiments, outcomes, outcomes_to_show=outcome, density=plotting.Density.BOXENPLOT, group_by='vd_switch', grouping_specifiers= grouping_specifiers_VD,legend=True)
    

    fig.set_size_inches(25, 10)
    
    repeat_token = 1
    fig.savefig(os.path.join(fig_path,str(run) + '_v6_OE_Time_grouped_' + str(outcome) + '_repeat_' +str(repeat_token) + '.png'))
plt.show()


# %%
outcome = 'Disutility of Damage'
fig,axes=plotting.envelopes(experiments, outcomes, outcomes_to_show=outcome, density=plotting.Density.BOXENPLOT, group_by='vd_switch', grouping_specifiers= grouping_specifiers_VD,legend=True)
    

fig.set_size_inches(25, 10)
    
repeat_token = 1
fig.savefig(os.path.join(fig_path, str(run) + '1a_Time_grouped_' + str(outcome) + '_repeat' +str(repeat_token) + '.png'))
plt.show()


# %%
out_DF = pd.DataFrame(outcomes)
out_DF.to_csv('v5_output.csv')


# %%