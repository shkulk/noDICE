# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%%
import os

myfolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(myfolder)
# %%
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

from dicemodel.noDICE_v4 import PyDICE
# from noDICE_v4 import PyDICE
model_version = 'v4'


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
                             IntegerParameter('t2xco2_dist',0,2),
                             IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                            #  IntegerParameter('cback', 100, 600)
                            ]
    
    dice_sm.levers = [#RealParameter('sr', 0.2, 0.3),
                    #   RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('prtp_dam',  0.001, 0.015),
                    #   RealParameter('emuc',  1.01, 2.00),
                      RealParameter('emdd', -1.0, 0.99),
                      IntegerParameter('periodfullpart', 10, 58),
                    #   IntegerParameter('miu_period', 10, 58)
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
    dice_sm.outcomes = [TimeSeriesOutcome('Atmospheric Temperature'),
                        # TimeSeriesOutcome('Per Capita Consumption'),
                        TimeSeriesOutcome('Consumption Growth'),
                        TimeSeriesOutcome('Utility of Consumption'),
                        # TimeSeriesOutcome('Per Capita Damage'),
                        # TimeSeriesOutcome('Damage Growth'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        TimeSeriesOutcome('Welfare'),
                        TimeSeriesOutcome('Undiscounted Period Welfare'),
                        TimeSeriesOutcome('Consumption SDR'),
                        # TimeSeriesOutcome('Damage SDR')

                        ]


# %%
n_scenarios = 1000
n_policies = 20

# %%
start = time.time()
with SequentialEvaluator(dice_sm) as evaluator:
    results = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)
end = time.time()

print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')


# %%
start = time.time()
with MultiprocessingEvaluator(dice_sm, n_processes=8) as evaluator:
    results = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)
end = time.time()

print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')

experiments, outcomes = results


# %%

save_results(results, os.path.join(gz_path, 'Nordhaus_OE' + str(n_scenarios) + 's_' + str(n_policies) + 'p_' + '.tar.gz'))


# %%
results = load_results(os.path.join(gz_path,'1a_OE_1000s_20p_.tar.gz'))
experiments, outcomes = results
# outcomes


# %%
## pairs scatter plot: grouped by policy

fig, axes = ema_workbench.analysis.pairs_plotting.pairs_scatter(experiments, outcomes)
fig.set_size_inches(50,50)
plt.show()

# %%
fig.savefig(os.path.join(fig_path,'Nordhaus_OE' + str(n_scenarios) + 's' + str(n_policies) + 'p' + '.png'))

# %%
grouping_specifiers = {'a':1, 'b':2, 'c':3}
grouping_labels = sorted(grouping_specifiers.keys())
grouping_specifiers = [grouping_specifiers[key] for key in
                                       grouping_labels]
# grouping_specifiers
low = np.min(experiments['emdd'])
high = np.max(experiments['emdd'])
grouping_specifiers_emdd = {'Low': low, 'mid': 0, 'High': high}


# %%
## pairs plotting by emdd range intervals
fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='emdd',grouping_specifiers=grouping_specifiers_emdd, legend=True, transparent=True, papertype='letter')
fig.set_size_inches(50,50)
plt.show()

# %%
fig.savefig(os.path.join(fig_path,'Nordhaus_pairplot_grouped_by_emdd' + str(n_scenarios) + 's' + str(n_policies) + 'p' + '.png'))

# %%
TimeLine = []
for i in range(65):
    TimeLine.append(2010+i*5)
# TimeLine
# outcomes["TIME"] = np.array([TimeLine])
# outcomes

# %%
plt.close('all')

# %%
# sns.set_style("whitegrid")

for outcome in outcomes.keys():
    fig,axes=plotting.lines(experiments, outcomes, outcomes_to_show=outcome, density=plotting.Density.BOXENPLOT, legend=True)
    # fig.tight_layout()
    fig.set_size_inches(15,10)
    fig.savefig(os.path.join(fig_path,'Nordhaus_TimeSeries' + str(outcome) + '.png'))
plt.show()

#%%
# %%
np.mean(np.mean(outcomes['Damage Growth'], axis =1))


# %%
# results = load_results(os.path.join(gz_path,'v4_fdamage_1a_OE_sr100s_20p_.tar.gz'))
# experiments, outcomes = results

# %%
# if we want the end value
end_outcome = {}
for key, value in outcomes.items():
    end_outcome[key] = value[58]  
# end_outcome['TIME']
# np.mean(end_outcome['Damage Growth'], axis =0)
# end_outcome
#%%
rem_list =['Total Output', 'Per Capita Consumption', 'Per Capita Damage','Damage SDR']
for key in rem_list:
    end_outcome.pop(key) 

# %%
out_DF = pd.DataFrame(end_outcome)
out_DF.head()
emdd = experiments['emdd']
out_DF['emdd'] = emdd

low = -0.5
# %%
out_DF['emdd'] = out_DF['emdd'].apply(lambda x: '-0.99 to -0.5' if x < -0.5 else ('-0.5 to 0' if x==1 else '0 to 0.99'))
#%%
# RealParameter('emdd', -1.0, 0.99)
# Need to run this only as single period output, otherwise it is difficult to match experiments (emdd values) with outcomes
out_DF['emdd'] = out_DF['emdd'].apply(lambda x: '-0.99 to -0.5' if x < -0.5 else ('-0.5 to 0' if(x < 0) else ('0 to 0.5' if (x < 0.5) else '0.5 to 0.99') ))
# %%
# out_DF = out_DF.drop(columns=['Total Output', 'Per Capita Consumption', 'Per Capita Damage','Damage SDR' ])

# %%
clr_palette = ([sns.cubehelix_palette(8)[6],sns.color_palette("inferno", 15)[-2],sns.color_palette("YlGn", 15)[10]])


# %%
sns.set_style("whitegrid")
sns_plot = sns.pairplot(out_DF, hue='emdd', palette=clr_palette, vars=list(end_outcome.keys()), height=2)
fig.set_size_inches(30, 30)
sns_plot.savefig(os.path.join(fig_path,'Nordhaus_sns_emdd' +'.png'))
plt.show()


# %%
# sns.set_style("whitegrid")
# for outcome in outcomes.keys():
#     # colmlst = list(range(0+_*4,4+_*4))
#     # namelist = colmlst[:]
#     # colmlst[2] = -3
#     # namelist[2] = -2
#     # ppDF = out_DF.iloc[:,colmlst+[-1]]
#     sns_plot = sns.pairplot(out_DF, hue='emdd', palette=clr_palette, vars=list(end_outcome.keys()))
#     plt.show()
#     # sns_plot.savefig((fig_path + 'dist_pairplot_2330' + str(end_outcome.keys()) +'_V4.png'), bbox_inches=false)
#     sns_plot.savefig(os.path.join(fig_path,'v4_dist_pairplot_2330' + str(end_outcome.keys()) +'.png'))
#     # sns_plot.savefig(fig_path + 'dist_pairplot_'+str(list(outcomes.keys())[_*4][-4:]) +'_V4.png')
#     # sns_plot.savefig(os.path.join(fig_path,'v4_fdamage_1a_TimeSeries_' + str(outcome) + str(n_scenarios) +'s' + str(n_policies) + 'p' + '.png')), bbox_inches="tight")

