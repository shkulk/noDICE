
# %%
from IPython import get_ipython
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair
# sns.set_style('white')
# import statsmodels.api as sm
from ema_workbench import (Model, RealParameter, IntegerParameter, ArrayOutcome, TimeSeriesOutcome,
                           ema_logging, SequentialEvaluator,
                           MultiprocessingEvaluator)
from ema_workbench import save_results, load_results
import ema_workbench.analysis
from ema_workbench.analysis import scenario_discovery_util as sdutil
from sklearn import preprocessing 
import ema_workbench.em_framework.evaluators
from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, ScalarOutcome, ArrayOutcome, Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)
# from ema_workbench.analysis import feature_scoring
# from ema_workbench.em_framework.salib_samplers import get_SALib_problem
# from SALib.analyze import sobol
  

ema_logging.log_to_stderr(ema_logging.INFO)
import os
import sys
os.chdir(os.getcwd())
pydice_folder = os.path.dirname(os.getcwd()) + "\\dicemodel"
sys.path.insert(1, pydice_folder)
pydice_folder

from noDICE_v3 import PyDICE

PyDICE()
#%%

# #%%
# if __name__ == '__main__':
#     ema_logging.log_to_stderr(ema_logging.INFO)
#     PyDICE(model_specification="EMA_disutility")


# # %%
run = PyDICE(model_specification="EMA_disutility")