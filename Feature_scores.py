# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# GSA with feature scoring


# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from ema_workbench import ema_logging, load_results
from ema_workbench.analysis import (get_ex_feature_scores,
                                    RuleInductionType)

ema_logging.log_to_stderr(level=ema_logging.INFO)
# from dicemodel.noDICE_v3 import PyDICE
# model_version = 'v4'

import os
from dest_directories import gz_path, fig_path

#%%
# n_scenarios = 2000
# n_policies = 50
# run = '36_OE'

n_scenarios = 100000
n_policies = 1
run = 'run_35_NordOE'


# %%
results = load_results(os.path.join(gz_path,'run_35_NordOE_v7_100000s_.tar.gz'))

#%%
experiments, outcomes = results
#%%
experiments = experiments.drop(['model', 'policy'], axis=1)


# %%
TimeLine = []
for i in range(65):
    TimeLine.append(2020+i*5)
TimeLine
outcomes["TIME"] = np.array([TimeLine])
# outcomes


# %%

for key, value in outcomes.items():
    outcomes[key] = value[:,2:-5]  
outcomes['Welfare'].shape

# %%
#  Welfare
from ema_workbench.analysis import feature_scoring

y = outcomes['Welfare']
all_scores = []

# we only want to show those uncertainties that are in the top 5
# most sensitive parameters at any time step
top_5 = set()
for i in range(2, y.shape[1], 8):
    data = y[:, i]
    scores = get_ex_feature_scores(experiments, data,
                                   mode=RuleInductionType.REGRESSION)[0]
    # add the top five for this time step to the set of top5s
    top_5 |= set(scores.nlargest(5, 1).index.values)   
    scores = scores.rename(columns={1:outcomes['TIME'][0, i]})
    all_scores.append(scores)
    
all_scores = pd.concat(all_scores, axis=1, sort=False)
all_scores = all_scores.loc[top_5, :]

fig, ax = plt.subplots()
sns.heatmap(all_scores, ax=ax, cmap='viridis', annot=True)
fig.set_size_inches(10,10)
plt.show()
take = 1
fig.savefig(os.path.join(fig_path,str(run) + '_Feature_scores_Wel' + str(n_scenarios) + 's_' + str(n_policies) + 'p_take_' + str(take) + '.png'))

# %%
# Total Output
y = outcomes['Total Output']
all_scores = []

# we only want to show those uncertainties that are in the top 5
# most sensitive parameters at any time step
top_5 = set()
for i in range(2, y.shape[1], 8):
    data = y[:, i]
    scores = get_ex_feature_scores(experiments, data,
                                   mode=RuleInductionType.REGRESSION)[0]
    # add the top five for this time step to the set of top5s
    top_5 |= set(scores.nlargest(5, 1).index.values)   
    scores = scores.rename(columns={1:outcomes['TIME'][0, i]})
    all_scores.append(scores)
    
all_scores = pd.concat(all_scores, axis=1, sort=False)
all_scores = all_scores.loc[top_5, :]

fig, ax = plt.subplots()
sns.heatmap(all_scores, ax=ax, cmap='viridis', annot=True)
fig.set_size_inches(10,10)
plt.show()
take = 3
fig.savefig(os.path.join(fig_path,str(run) + '_Feature_scores_Output' + str(n_scenarios) + 's_' + str(n_policies) + 'p_take_' + str(take) + '.png'))

# %%
#  Utility of Consumption
y = outcomes['Utility of Consumption']
all_scores = []

# we only want to show those uncertainties that are in the top 5
# most sensitive parameters at any time step
top_5 = set()
for i in range(2, y.shape[1], 8):
    data = y[:, i]
    scores = get_ex_feature_scores(experiments, data,
                                   mode=RuleInductionType.REGRESSION)[0]
    # add the top five for this time step to the set of top5s
    top_5 |= set(scores.nlargest(5, 1).index.values)   
    scores = scores.rename(columns={1:outcomes['TIME'][0, i]})
    all_scores.append(scores)
    
all_scores = pd.concat(all_scores, axis=1, sort=False)
all_scores = all_scores.loc[top_5, :]

fig, ax = plt.subplots()
sns.heatmap(all_scores, ax=ax, cmap='viridis', annot=True)
fig.set_size_inches(10,10)
plt.show()

take = 1
fig.savefig(os.path.join(fig_path,str(run) + '_Feature_scores_UtilC' + str(n_scenarios) + 's_' + str(n_policies) + 'p_take_' + str(take) + '.png'))

# %%
#  Disutility of Damage
y = outcomes['Disutility of Damage']
all_scores = []

# we only want to show those uncertainties that are in the top 5
# most sensitive parameters at any time step
top_5 = set()
for i in range(2, y.shape[1], 8):
    data = y[:, i]
    scores = get_ex_feature_scores(experiments, data,
                                   mode=RuleInductionType.REGRESSION)[0]
    # add the top five for this time step to the set of top5s
    top_5 |= set(scores.nlargest(5, 1).index.values)   
    scores = scores.rename(columns={1:outcomes['TIME'][0, i]})
    all_scores.append(scores)
    
all_scores = pd.concat(all_scores, axis=1, sort=False)
all_scores = all_scores.loc[top_5, :]

fig, ax = plt.subplots()
sns.heatmap(all_scores, ax=ax, cmap='viridis', annot=True)
fig.set_size_inches(10,10)
plt.show()

take = 1
fig.savefig(os.path.join(fig_path,str(run) + '_Feature_scores_VD' + str(n_scenarios) + 's_' + str(n_policies) + 'p_take_' + str(take) + '.png'))

# %%
# %%
#  Temperature
y = outcomes['Atmospheric Temperature']
all_scores = []

# we only want to show those uncertainties that are in the top 5
# most sensitive parameters at any time step
top_5 = set()
for i in range(2, y.shape[1], 8):
    data = y[:, i]
    scores = get_ex_feature_scores(experiments, data,
                                   mode=RuleInductionType.REGRESSION)[0]
    # add the top five for this time step to the set of top5s
    top_5 |= set(scores.nlargest(5, 1).index.values)   
    scores = scores.rename(columns={1:outcomes['TIME'][0, i]})
    all_scores.append(scores)
    
all_scores = pd.concat(all_scores, axis=1, sort=False)
all_scores = all_scores.loc[top_5, :]

fig, ax = plt.subplots()
sns.heatmap(all_scores, ax=ax, cmap='viridis', annot=True)
fig.set_size_inches(10,10)
plt.show()

take = 1
fig.savefig(os.path.join(fig_path,str(run) + '_Feature_scores_Temp' + str(n_scenarios) + 's_' + str(n_policies) + 'p_take_' + str(take) + '.png'))

# %%
