# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import time
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
# import statsmodels.api as sm

from ema_workbench import (Model, RealParameter, IntegerParameter, ArrayOutcome, TimeSeriesOutcome,
                           ema_logging, SequentialEvaluator,
                           MultiprocessingEvaluator)
from ema_workbench import save_results, load_results
# from ema_workbench.analysis import prim, cart
from ema_workbench.analysis import scenario_discovery_util as sdutil
from sklearn import preprocessing 
from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS
# from ema_workbench.analysis import feature_scoring
# from ema_workbench.em_framework.salib_samplers import get_SALib_problem
# from SALib.analyze import sobol


ema_logging.log_to_stderr(ema_logging.INFO)
from dicemodel.noDICE_v2 import PyDICE


# %%
from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, ScalarOutcome, ArrayOutcome, 
                           Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)

# %% [markdown]
# # Time series 

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
                    #   RealParameter('emuc',  0.01, 2.00),
                      RealParameter('emdd',  0.01, 2.00),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]
    
    dice_sm.outcomes = [TimeSeriesOutcome('Atmospheric Temperature'),
                        TimeSeriesOutcome('Damages'),
                        TimeSeriesOutcome('Utility of Consumption'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        # TimeSeriesOutcome('Damage to output ratio'),
                        TimeSeriesOutcome('Welfare'),
                        TimeSeriesOutcome('Total Output'),
                        TimeSeriesOutcome('Consumption SDR'),
                        TimeSeriesOutcome('Damage SDR')
                        ]
    
    
    # n_scenarios = 2
    # n_policies = 1


# %%
n_scenarios = 100
n_policies = 50


# %%
start = time.time()
with MultiprocessingEvaluator(dice_sm, n_processes=8) as evaluator:
    results2 = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)
end = time.time()

print('Experiment time is ' + str(round((end - start)/60)) + ' mintues')


# %%
experiments, outcomes = results2

from ema_workbench.analysis import plotting, plotting_util


# %%
# pydice_folder = os.path.dirname(os.getcwd())+"\\1_Model"
# sys.path.insert(1, pydice_folder)
# file_name = 'exploration_V4_' + str(n_scenarios) + 'scen_' + 'nordhaus_optimal_policy_' + str(4) + 'obj' + '.tar.gz'
# #save_results(results, file_name)
results = results2
save_results(results, '1a_OE_100p_50s.tar.gz')


# %%
load_results('1a_OE_100p_50s.tar.gz')


# %%
## pairs plotting
from ema_workbench.analysis import pairs_plotting
fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='policy',legend=True)
fig.set_size_inches(20,20)
plt.show()


# %%
fig.savefig('1a_OE_pairsplot' + str(n_scenarios) + 's' + str(n_policies) + 'p' + '.png')


# %%
## pairs plotting 100p, 50s
from ema_workbench.analysis import pairs_plotting
fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes,legend=True)
fig.set_size_inches(20,20)
plt.show()


# %%

sns.set_style("whitegrid")

for outcome in outcomes.keys():
    fig,axes=plotting.lines(experiments, outcomes, outcomes_to_show=outcome, density=plotting.Density.HIST)
    fig.set_size_inches(12, 5)
    fig.savefig('1a_TimeSeries' + str(n_scenarios) +'s' + str(n_policies) + 'p' + '.png')
plt.show()


# %%
for outcome in outcomes.keys():
    fig, axis = sns.boxenplot(x=outcomes[tstep])


# %%
# keeping EMUC constant, 10p 10s
sns.set_style("whitegrid")

for outcome in outcomes.keys():
    fig,axes=plotting.lines(experiments, outcomes, outcomes_to_show=outcome, density=plotting_util.Density.BOXPLOT)
    fig.set_size_inches(12, 5)
    # fig.savefig('Time_Series_Plot_V4_' + str(outcome) + '_scen' + str(n_scenarios) + '_pol' + str(n_policies) + '.png')
plt.show()


# %%
## pairs plotting
from ema_workbench.analysis import pairs_plotting
fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='policy',legend=True)
fig.set_size_inches(20,20)
plt.show()


# %%
## pairs plotting
from ema_workbench.analysis import pairs_plotting
fig, axes = pairs_plotting.pairs_scatter(experiments,outcomes, group_by='policy',legend=True)
fig.set_size_inches(20,20)
plt.show()


# %%



# %%


