if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import time

    import os
    os.chdir(os.getcwd())
    import sys
    # insert at 1, 0 is the script path (or '' in REPL)
    # pydice_folder = os.path.dirname(os.getcwd())
    # sys.path.insert(1, pydice_folder)

    from specify import (specify_levers, specify_scenario)

    from ema_workbench import (Model, Policy, Scenario, RealParameter, IntegerParameter, ScalarOutcome, MultiprocessingEvaluator)
    from ema_workbench.em_framework.evaluators import perform_experiments
    from ema_workbench.em_framework.samplers import sample_uncertainties
    from ema_workbench.util import ema_logging

    from ema_workbench.em_framework.evaluators import BaseEvaluator
    from ema_workbench.em_framework.optimization import EpsilonProgress

    ema_logging.log_to_stderr(ema_logging.INFO)
    BaseEvaluator.reporting_frequency = 0.1

    # from PyDICE_V4 import PyDICE
    from PyDICE_190620_2022 import PyDICE
    model = PyDICE()
    dice_sm = Model('dicesmEMA', function = model)
    dice_opt = pd.read_excel("DICE2013R.xlsm" ,sheet_name = "Opttax", index_col = 0)
    print("XLRM", flush=True)

    dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                             IntegerParameter('t2xco2_dist',0 , 2),
                             IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr', 0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]

    dice_sm.levers = [RealParameter('sr', 0.1, 0.5),
                      RealParameter('irstp',  0.001, 0.015),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]

    dice_sm.outcomes = [ScalarOutcome('Atmospheric Temperature 2050', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2050', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2050', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2050', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2100', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2100', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2100', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2100', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2150', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2150', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2150', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2150', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2200', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2200', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2200', ScalarOutcome.INFO),
                        ScalarOutcome('Total Output 2200', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Atmospheric Temperature 2300', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Damages 2300', ScalarOutcome.MINIMIZE),
                        ScalarOutcome('Utility 2300', ScalarOutcome.MAXIMIZE),
                        ScalarOutcome('Total Output 2300', ScalarOutcome.MAXIMIZE)]

    eps = [0.001, 0.1, 0.1, 0.1] * (int(len(dice_sm.outcomes)/4.0))
    convergence_metrics = [EpsilonProgress()]
    nord_optimal_policy = Policy('nord_optimal_policy', **specify_levers(np.mean(dice_opt.iloc[129]), 0.015, 0, 29))
    nfe = 100000
    # Swtich to the worst case
    for outcome in dice_sm.outcomes:
        if outcome.kind == ScalarOutcome.MINIMIZE:
            outcome.kind = ScalarOutcome.MAXIMIZE
        else:
            outcome.kind = ScalarOutcome.MINIMIZE

    start = time.time()
    print("starting search for wcs", flush=True)
    with MultiprocessingEvaluator(dice_sm, n_processes=16) as evaluator:
        results, convergence = evaluator.optimize(nfe=nfe,
                                                  searchover='uncertainties',
                                                  reference=nord_optimal_policy,
                                                  epsilons=eps,
                                                  convergence=convergence_metrics
                                                  )
    results.to_csv("wcs_100k_nfe_V4.csv")
    convergence.to_csv("wcs_con_100k_nfe_V4.csv")
    end = time.time()
    print('Directed Search of WCS time is ' + str(round((end - start) / 60)) + ' mintues')
