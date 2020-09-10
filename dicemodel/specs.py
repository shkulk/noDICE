from ema_workbench import Scenario

def nordhaus_policy(sr = 0, prtp_con = 0,periodfullpart =  0, miu_period = 0, emuc = 0) : #prtp_dam=0, vd_switch = 0
    
    '''Specify levers for further usage'''
    
    levers_param = {'sr': sr, 'prtp con': prtp_con, 'emuc': emuc, 'periodfullpart': periodfullpart, 'miu_period': miu_period} # 'vd_switch': vd_switch,
    
    return levers_param

def reference_scenario(reference_values, dice_sm) :
    scen = {}
    for key in dice_sm.uncertainties:
        scen.update({key.name: reference_values[key.name]})
    reference_scenario = Scenario('reference', **scen)
    return reference_scenario

def change_fontsize(fig, fs=14):
    '''Change fontsize of figure items to specified size'''

    for ax in fig.axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                      ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs)
        
        try:
            parasites = ax.parasites
        except AttributeError:
            pass
        else:
            for parisite in parasites:
                for axis in parisite.axis.values():
                    axis.major_ticklabels.set_fontsiz(fs)
                    axis.label.set_fontsize(fs)
                for axis in ax.axis.values():
                    axis.major_ticklabels.set_fontsiz(fs)
                    axis.label.set_fontsize(fs)
			
                if ax.legend_ != None:
                    for entry in ax.legend_.get_texts():
                        entry.set_fontsize(fs)
                
                for entry in ax.texts:
                    entry.set_fontsize(fs)
                    
                for entry in ax.tables:
                    entry.set_fontsize(fs)

# def default_scenario(dike_model) : 
#     reference_values = {'Bmax': 175, 'Brate': 1.5, 'pfail': 0.5,
#                         'discount rate': 3.5,
#                         'ID flood wave shape': 4}
#     scen1 = {}

#     for key in dike_model.uncertainties:
#         name_split = key.name.split('_')

#         if len(name_split) == 1:
#             scen1.update({key.name: reference_values[key.name]})

#         else:
#             scen1.update({key.name: reference_values[name_split[1]]})

#     ref_scenario = Scenario('reference', **scen1)
#     return ref_scenario
