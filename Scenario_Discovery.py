#%%
import time
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
from dicemodel.noDICE_v6 import PyDICE
model_version = 'v6'

from dicemodel.specs import nordhaus_policy, reference_scenario

# %%
from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, TimeSeriesOutcome, ScalarOutcome, ArrayOutcome, Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)
from ema_workbench import save_results, load_results
from ema_workbench.analysis import prim, cart
from ema_workbench.analysis import scenario_discovery_util as sdutil
from ema_workbench.analysis import plotting, plotting_util

ema_logging.log_to_stderr(ema_logging.INFO)

model = PyDICE()
dice_sm = Model('dicesmEMA', function=model)

#%%
dice_opt = pd.read_excel("DICE2013R.xlsm", sheet_name="Opttax", index_col=0)



#%%
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
            ScalarOutcome('Atmospheric Temperature 2300', kind=ScalarOutcome.MINIMIZE),
            ScalarOutcome('Total Output 2300', kind=ScalarOutcome.MAXIMIZE),
            # ScalarOutcome('Per Capita Consumption 2300', kind=ScalarOutcome.INFO),
            # ScalarOutcome('Consumption Growth 2300', kind=ScalarOutcome.INFO),
            ScalarOutcome('Utility of Consumption 2300', kind=ScalarOutcome.MAXIMIZE),
            # ScalarOutcome('Per Capita Damage 2300', kind=ScalarOutcome.INFO),
            # ScalarOutcome('Damage Growth 2300', kind=ScalarOutcome.INFO),
            ScalarOutcome('Disutility of Damage 2300', kind=ScalarOutcome.MINIMIZE),
            ScalarOutcome('Welfare 2300', kind=ScalarOutcome.MAXIMIZE),
            # ScalarOutcome('Undiscounted Period Welfare 2300', kind=ScalarOutcome.INFO),
            ScalarOutcome('Consumption SDR 2300', kind=ScalarOutcome.INFO),
            ScalarOutcome('Damage SDR 2300', kind=ScalarOutcome.INFO)
            ]


# %%
n_scenarios = 1000
n_policies = 25
run = 10
#%%

# %%
with SequentialEvaluator(dice_sm) as evaluator:
    results1 = evaluator.optimize(nfe=5e3, searchover='levers', epsilons=[0.01,]*len(dice_sm.outcomes))
# %%
with SequentialEvaluator(dice_sm) as evaluator:
    results2 = evaluator.optimize(nfe=5e3, searchover='levers', epsilons=[0.1,]*len(dice_sm.outcomes))


# %%
from ema_workbench.analysis import parcoords
# import changefont as cf

data = results1.loc[:, [o.name for o in dice_sm.outcomes]]
limits = parcoords.get_limits(data)
# limits['Welfare 2300'] = [-18000,0]
# limits['Undiscounted Period Welfare 2300'] = [-1000, 0]
# limits['Consumption SDR 2300'] = [0, 0.1]
# limits['Damage SDR 2300'] = [0,0.5]
# manual_limits = pd.DataFrame(
    # {'Welfare 2300':[-18000,0],
    # 'Undiscounted Period Welfare 2300':[-1000,0],
    # 'Consumption SDR 2300':[0, 0.1],
    # 'Damage SDR 2300':[0,0.5],     
    # })

paraxes = parcoords.ParallelAxes(limits, rot=0)
paraxes.plot(data)
paraxes.fig.set_size_inches(25, 10)
paraxes.legend()
plt.show()
paraxes.fig.savefig(os.path.join(fig_path, str(run) + '_v6__sc_disc_par_results1' + '.png'))
#%%
data = results2.loc[:, [o.name for o in dice_sm.outcomes]]
limits = parcoords.get_limits(data)
# limits['Welfare 2300'] = [-18000,0]
# limits['Undiscounted Period Welfare 2300'] = [-1000, 0]
# limits['Consumption SDR 2300'] = [0, 0.1]
# limits['Damage SDR 2300'] = [0,0.5]
# manual_limits = pd.DataFrame(
    # {'Welfare 2300':[-18000,0],
    # 'Undiscounted Period Welfare 2300':[-1000,0],
    # 'Consumption SDR 2300':[0, 0.1],
    # 'Damage SDR 2300':[0,0.5],     
    # })

paraxes = parcoords.ParallelAxes(limits, rot=0)
paraxes.plot(data)
paraxes.fig.set_size_inches(25, 10)
paraxes.legend()
plt.show()
paraxes.fig.savefig(os.path.join(fig_path, str(run) + '_v6__sc_disc_par_results2' + '.png'))

# %%
from ema_workbench.em_framework.optimization import (HyperVolume,EpsilonProgress)

convergence_metrics = [HyperVolume(minimum=[0,0,0,0], maximum=[3, 2,1.01,1.01]),
                       EpsilonProgress()]

with MultiprocessingEvaluator(model) as evaluator:
    results, convergence = evaluator.optimize(nfe=1e3, searchover='levers',
                                 convergence=convergence_metrics,
                                 epsilons=[0.1,]*len(model.outcomes))

#%%
from dicemodel.specs import nordhaus_policy, reference_scenario
eps = [0.001, 0.1, 0.1, 0.1] * (int(len(dice_sm.outcomes)/4.0))
convergence_metrics = [EpsilonProgress()]
nord_optimal_policy = Policy('nord_optimal_policy', **nordhaus_policy(np.mean(dice_opt.iloc[129]), 0.015, 0, 0, 29))
nfe = 100000
# Swtich to the worst case
for outcome in dice_sm.outcomes:
    if outcome.kind == ScalarOutcome.MINIMIZE:
        outcome.kind = ScalarOutcome.MAXIMIZE
    else:
        outcome.kind = ScalarOutcome.MINIMIZE
#%%
start = time.time()
print("starting search for wcs", flush=True)
with MultiprocessingEvaluator(dice_sm, n_processes=16) as evaluator:
    results, convergence = evaluator.optimize(nfe=nfe,
                                                searchover='uncertainties',
                                                reference=nord_optimal_policy,
                                                epsilons=eps,
                                                convergence=convergence_metrics
                                                )
end = time.time()
results.to_csv( os.path.join(gz_path, str(run) + str(model_version)+"wcs_100k_nfe_V4.csv")
convergence.to_csv("wcs_con_100k_nfe_V4.csv")
end = time.time()
print('Directed Search of WCS time is ' + str(round((end - start) / 60)) + ' mintues')
