from dicemodel.noDICE_v3 import PyDICE
model_version = 'v3.4'
from ema_workbench import (perform_experiments, Model, Policy, Scenario, ReplicatorModel, RealParameter, IntegerParameter, TimeSeriesOutcome, ScalarOutcome, ArrayOutcome, Constant, ema_logging, SequentialEvaluator, MultiprocessingEvaluator, IpyparallelEvaluator)
from ema_workbench.analysis import pairs_plotting, plotting_util, plotting
model = PyDICE()
dice_sm = Model('dicesmEMA', function=model)

def exp_spec(exp_spec_name):
    
    if exp_spec_name == 'DICE2013': #DICE 2013 with endogenous discount rate
        # fdamage, t2xco2 OFF
        # All lever ranges except period full participation OFF: sr, prtp_con, prtp_dam, emuc, emdd

        dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                            #  IntegerParameter('t2xco2_dist',0,2),
                            #  IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]
    
        dice_sm.levers = [#RealParameter('sr', 0.1, 0.3),
                      # RealParameter('prtp_con',  0.001, 0.015),
                      # RealParameter('prtp_dam',  0.001, 0.015),
                    #   RealParameter('emuc',  1.01, 2.00),
                      # RealParameter('emdd',  1.01, 2.00),
                      IntegerParameter('periodfullpart', 10, 58),
                    #   IntegerParameter('miu_period', 10, 58)
                    ]
    elif exp_spec_name == 'VD_fixed_EMUC': # DICE2013 + V(D) approach
        # fdamage OFF, t2xco2 OFF
        # prtp_con, prtp_dam, emuc, emdd ON


        dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                            #  IntegerParameter('t2xco2_dist',0,2),
                            #  IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]
    
        dice_sm.levers = [RealParameter('sr', 0.1, 0.3),
                      RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  1.01, 2.00),
                      RealParameter('emdd',  1.01, 2.00),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]
    

    elif exp_spec_name == 'VD': # DICE2013 + V(D) approach
        # fdamage OFF, t2xco2 OFF
        # prtp_con, prtp_dam, emuc, emdd ON


        dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                            #  IntegerParameter('t2xco2_dist',0,2),
                            #  IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]
    
        dice_sm.levers = [RealParameter('sr', 0.1, 0.3),
                      RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  1.01, 2.00),
                      RealParameter('emdd',  1.01, 2.00),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]

    elif exp_spec_name == 'DamageFunc': # DICE2013 + V(D) approach + damage functions
        # fdamage ON, t2xco2 OFF
        # prtp_con, prtp_dam, emuc, emdd ON


        dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                            #  IntegerParameter('t2xco2_dist',0,2),
                             IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]
    
        dice_sm.levers = [RealParameter('sr', 0.1, 0.3),
                      RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  1.01, 2.00),
                      RealParameter('emdd',  1.01, 2.00),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]
    
    elif exp_spec_name == 'ECSdist': # DICE2013 + V(D) approach +damage functions + ECS distributions
        # fdamage ON, t2xco2 ON
        # prtp_con, prtp_dam, emuc, emdd ON


        dice_sm.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                            #  IntegerParameter('t2xco2_dist',0,2),
                            #  IntegerParameter('fdamage', 0, 2),
                             RealParameter('tfp_gr',  0.07, 0.09),
                             RealParameter('sigma_gr', -0.012, -0.008),
                             RealParameter('pop_gr', 0.1, 0.15),
                             RealParameter('fosslim',  4000.0, 13649),
                             IntegerParameter('cback', 100, 600)]
    
        dice_sm.levers = [RealParameter('sr', 0.1, 0.3),
                      RealParameter('prtp_con',  0.001, 0.015),
                      RealParameter('prtp_dam',  0.001, 0.015),
                      RealParameter('emuc',  1.01, 2.00),
                      RealParameter('emdd',  1.01, 2.00),
                      IntegerParameter('periodfullpart', 10, 58),
                      IntegerParameter('miu_period', 10, 58)]
    

    return dice_sm.uncertainties, dice_sm.levers
    
def outcome_spec(out_spec_name):
    
    if out_spec_name == 'all':
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
    
    elif out_spec_name == 'VD':
                dice_sm.outcomes = [TimeSeriesOutcome('Atmospheric Temperature'),
                        # TimeSeriesOutcome('Total Output'),
                        # TimeSeriesOutcome('Population'),
                        # TimeSeriesOutcome('Per Capita Consumption'),
                        # TimeSeriesOutcome('Consumption Growth'),
                        # TimeSeriesOutcome('Utility of Consumption'),
                        TimeSeriesOutcome('Per Capita Damage'),
                        TimeSeriesOutcome('Damage Growth'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        TimeSeriesOutcome('Welfare'),
                        TimeSeriesOutcome('Undiscounted Period Welfare'),
                        # TimeSeriesOutcome('Consumption SDR'),
                        TimeSeriesOutcome('Damage SDR')

                        ]

    elif out_spec_name == 'UC':
                dice_sm.outcomes = [TimeSeriesOutcome('Atmospheric Temperature'),
                        # TimeSeriesOutcome('Total Output'),
                        # TimeSeriesOutcome('Population'),
                        TimeSeriesOutcome('Per Capita Consumption'),
                        TimeSeriesOutcome('Consumption Growth'),
                        TimeSeriesOutcome('Utility of Consumption'),
                        TimeSeriesOutcome('Per Capita Damage'),
                        # TimeSeriesOutcome('Damage Growth'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        TimeSeriesOutcome('Welfare'),
                        TimeSeriesOutcome('Undiscounted Period Welfare'),
                        TimeSeriesOutcome('Consumption SDR'),
                        # TimeSeriesOutcome('Damage SDR')

                        ]

    elif out_spec_name == 'light':
                dice_sm.outcomes = [#TimeSeriesOutcome('Atmospheric Temperature'),
                        # TimeSeriesOutcome('Total Output'),
                        # TimeSeriesOutcome('Population'),
                        TimeSeriesOutcome('Per Capita Consumption'),
                        # TimeSeriesOutcome('Consumption Growth'),
                        TimeSeriesOutcome('Utility of Consumption'),
                        TimeSeriesOutcome('Per Capita Damage'),
                        # TimeSeriesOutcome('Damage Growth'),
                        TimeSeriesOutcome('Disutility of Damage'),
                        TimeSeriesOutcome('Welfare'),
                        TimeSeriesOutcome('Undiscounted Period Welfare'),
                        # TimeSeriesOutcome('Consumption SDR'),
                        # TimeSeriesOutcome('Damage SDR')

                        ]
    
    # elif out_spec_name == 'final':
    #     dice_sm.outcomes = [
    #         ScalarOutcome('Atmospheric Temperature 2300'),
    #         ScalarOutcome('Per Capita Consumption 2300'),
    #         ScalarOutcome('Utility of Consumption 2300'),
    #         ScalarOutcome('Per Capita Damage 2300'),
    #         ScalarOutcome('Disutility of Damage 2300'),
    #         ScalarOutcome('Welfare 2300'),
    #         ScalarOutcome('Total Output 2300')            

        # ]
    
    return dice_sm.outcomes
    