# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# verify behaviour


# %%
import numpy as np
import pandas as pd

from dicemodel.noDICE_v3 import PyDICE


# %%
tstep = 5
t= 10

prtp_dam = 0.015
emdd = 1.45/2
dam_g = 0.16
dpc = 0.7/10

prtp_con = 0.015
emuc = 1.45
con_g = 0.09
cpc = 20/10


# %%
sdr_dam = (prtp_dam + (emdd * dam_g))
sdr_con = (prtp_con + (emuc * con_g))

# sdr_con = 0.015
# sdr_dam= sdr_con

print(sdr_dam, sdr_con)


# %%
# Social discount factor for disutility of damage            
damage_sdf = (1.0 /(1.0 + sdr_dam))**(tstep * (t))
consumption_sdf = (1.0 /(1.0 + sdr_con))**(tstep * (t))

# print(damage_sdf, consumption_sdf)


# %%
inst_disutil_dam = ((dpc ** (1.0 - emdd)-1.0)/ (1.0 - emdd)- 1.0)
# inst_disutil_dam = np.log(dpc)
inst_util_con = ((cpc ** (1.0 - emuc))/ (1.0 - emuc))


print(inst_disutil_dam, inst_util_con)
print(inst_util_con-inst_disutil_dam)

# print(inst_util_con+inst_disutil_dam)


# %%
uc_marginal = inst_util_con/cpc

vd_marginal = inst_disutil_dam/dpc
print(uc_marginal)

print(vd_marginal)


# %%
disc_disutil_dam = inst_disutil_dam * damage_sdf
disc_util_con = inst_util_con * consumption_sdf

print(disc_disutil_dam,disc_util_con)
print(disc_util_con - disc_disutil_dam)


# %%



# %%
prtp_con = 0.015
emuc = 1.45
con_g = 0.09
cpc = 200
sdr_con = (prtp_con + (emuc * con_g))
# print(sdr_con)
consumption_sdf = (1.0 /(1.0 + sdr_con))**(tstep * (t))
# print(consumption_sdf)
inst_util_con = ((cpc ** (1.0 - emuc))/ (1.0 - emuc))
print(inst_util_con)
print(inst_util_con-inst_disutil_dam)
disc_util_con = inst_util_con * consumption_sdf

# print(disc_util_con*1000)
# print((disc_util_con - disc_disutil_dam) * 1000)

#%%
# Paracoords
from ema_workbench.analysis import parcoords
# conditional on y_outcome

# What do I want here? a DF for each outcome, with all outcome values as dataframe rows, irrespective of experiment
# y['Welfare'] < -10

data = pd.DataFrame({k:v[y] for k,v in end_outcome.items()})
all_data = pd.DataFrame({k:v for k,v in outcomes.items()})

limits = parcoords.get_limits(all_data)
axes = parcoords.ParallelAxes(limits)
axes.plot(all_data, color='lightgrey')
axes.plot(data, color='blue')
# axes.invert_axis('max_P')
plt.show()



# %%
# if we want the end value (reducing across time)
end_outcome = {}
for key, value in outcomes.items():
    end_outcome[key] = value[58]  
# end_outcome['TIME']
# np.mean(end_outcome['Damage Growth'], axis =0)
# end_outcome

#  reucing across outcomes of interest ( dict keys)
rem_list =['Total Output', 'Per Capita Consumption', 'Per Capita Damage','Damage SDR']
for key in rem_list:
    end_outcome.pop(key)
    
del myDict['A'][1]
# myDict['A'] - It will grab the Value w.r.t Key A
# myDict['A'][1] - It will grab first index tuple
# del myDict['A'][1] - now this will delete that tuple

# or 
myDict['A'].pop(1)

# I want to remove first n = 5 tsteps 
y = outcomes
for n in range(1, 5):
     for key, value in y.items():
         y[key].pop(0)

# y["Welfare"].shape
# y['Welfare'].pop(0)
y['Welfare'][0]  # this is my result per point in time * each experiment, so I can use this to look at any point in time/ end time
# has shape (n_exp, )

y['Welfare'][65]
y['Welfare'] < -10
# %%
