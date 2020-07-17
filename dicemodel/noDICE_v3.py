# DICE - Model

# v3.3: corrected the cpc function
# IMPORT PACKAGES & SET PATH
import numpy as np
import pandas as pd
from scipy.stats import norm, skewnorm, cauchy, lognorm
import logging

import json
import os


myfolder = os.path.dirname(os.path.realpath(__file__))

class PyDICE(object):
    """ DICE simulation model:
        tstep: time step/interval
        steps: amount of years looking into the future
        model_specification: model specification for 'EMA_disutility' or 'Validation'  
    """
    def __init__(self, tstep=5, steps=60, model_specification="EMA_disutility"):
        self.tstep = tstep					# (in years)
        self.steps = steps
        self.startYear = 2010
        self.tperiod = []
        self.model_specification = model_specification
        
        # seems hopelesly inefficient 
        # why not use numpy arrays?
        for i in range(0, self.steps):
            self.tperiod.append((i*self.tstep)+self.startYear)


        with open(os.path.join(myfolder, 'ecs_dist_v4.json')) as f:
            d = json.load(f)
        
        #creating a list from the dist of t2xC02
        np.random.seed(10)

        minb = 0
        maxb = 20
        nsamples = 1000

        samples_norm = np.zeros((0,))
        while samples_norm.shape[0] < nsamples:
            samples = (norm.rvs(d['norm'][0], d['norm'][1],nsamples))
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_norm = np.concatenate((samples_norm, accepted), axis=0)
        samples_norm = samples_norm[:nsamples]

        samples_lognorm = np.zeros((0,))
        while samples_lognorm.shape[0] < nsamples:
            samples = (lognorm.rvs(d['lognorm'][0], d['lognorm'][1], 
                                   d['lognorm'][2],nsamples))
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_lognorm = np.concatenate((samples_lognorm, accepted), axis=0)
        samples_lognorm = samples_lognorm[:nsamples]

        samples_cauchy = np.zeros((0,))
        while samples_cauchy.shape[0] < nsamples:
            samples = (cauchy.rvs(d['cauchy'][0],d['cauchy'][1],nsamples))
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_cauchy = np.concatenate((samples_cauchy, accepted), axis=0)
        samples_cauchy = samples_cauchy[:nsamples]

        # extend array with the deterministic value of the nordhaus

        samples_norm = np.append(samples_norm, 2.9)
        samples_lognorm = np.append(samples_lognorm, 2.9)
        samples_cauchy = np.append(samples_cauchy, 2.9)
        
        self.samples_t2xco2 = [samples_norm, samples_lognorm, samples_cauchy]
        
    def __call__(self,
                 # uncertainties from Nordhaus(2008)
                 t2xco2_index=-1,
                 t2xco2_dist=0,
                 tfp_gr=0.079,
                 sigma_gr=-0.01, # growth rate sigma = Carbon intensity
                 pop_gr=0.134,
                 fosslim=6000,
                 cback=344,
                 decl_back_gr=0.025,
                 limmiu=1.2,
                 fdamage=0,
                 # levers from Nordhaus(2008) 
                 sr=0.249,
                 prtp_con = 0.015, 
                 prtp_dam = 0.015, 
                 emuc = 1.45, #from nordhaus
                 emdd = 1.45, # default equivalent to emuc to simulate Nordhaus
                 periodfullpart=21,
                 miu_period=29, #17
                 **kwargs):

        """
        ####################################################################
        ######################### MODEL-SET UPS ############################
        ####################################################################
        """
        """
        ####################### ECONOMICS PARAMETERS #######################
        """
        # time indexed variables declaration
        self.pop = np.zeros((self.steps,))
        self.tfp = np.zeros((self.steps,))
        self.tfp_gr = np.zeros((self.steps,))
        self.k = np.zeros((self.steps,))
        self.i = np.zeros((self.steps,))
        self.ygross = np.zeros((self.steps,))
        self.ynet = np.zeros((self.steps,))     #  output - damages
        self.damfrac = np.zeros((self.steps,))
        self.damages = np.zeros((self.steps,))  # damages = ygross* damfrac
        self.pbacktime = np.zeros((self.steps,))
        self.cost1 = np.zeros((self.steps,)) # self.backstop_cost (rename)
        self.partfract = np.zeros((self.steps,))
        self.abatecost = np.zeros((self.steps,))
        # self.miu_up = np.zeros((self.steps,)) # ECS upper limit
        self.sigma = np.zeros((self.steps,))
        self.sigma_gr = np.zeros((self.steps,))
        self.eind = np.zeros((self.steps,))
        self.cumetree = np.zeros((self.steps,))
        self.etree = np.zeros((self.steps,))
        self.e = np.zeros((self.steps,)) #total emission
        self.cca = np.zeros((self.steps,)) # cumulative emissions (rename?)
        self.y = np.zeros((self.steps,)) #this is actually income (rename?)
        self.c = np.zeros((self.steps,))
        self.cpc = np.zeros((self.steps,))
        self.dpc = np.zeros((self.steps,))
        self.con_g = np.zeros((self.steps,)) # consumption growth
        self.dam_g = np.zeros((self.steps,)) # damage growth
        self.sdr_con = np.zeros((self.steps,))
        self.sdr_dam = np.zeros((self.steps,))
        self.consumption_sdf = np.zeros((self.steps,))
        self.damage_sdf = np.zeros((self.steps,))
        self.inst_util_con = np.zeros((self.steps,))
        self.disc_util_con = np.zeros((self.steps,))
        self.inst_disutil_dam = np.zeros((self.steps,))
        self.disc_disutil_dam = np.zeros((self.steps,))
        self.period_welfare = np.zeros((self.steps,))
        self.welfare = np.zeros((self.steps,))
        self.cprice = np.zeros((self.steps,))
        self.ccatot = np.zeros((self.steps,))
        self.scc = np.zeros((self.steps,))
        self.atfrac = np.zeros((self.steps,))
        self.atfrac2010 = np.zeros((self.steps,))
        self.ppm = np.zeros((self.steps,))

        """
        ######################### CARBON PARAMETERS ########################
        """

        self.mat = np.zeros((self.steps,))
        self.mu = np.zeros((self.steps,))
        self.ml = np.zeros((self.steps,))
        self.forcoth = np.zeros((self.steps,))
        self.forc = np.zeros((self.steps,))

        """
        ######################## CLIMATE PARAMETERS ########################
        """

        # Increase temperature of atmosphere [dC from 1900]
        self.temp_atm = np.zeros((self.steps,))
        # Increase temperature of lower oceans [dC from 1900]
        self.temp_ocean = np.zeros((self.steps,))

        """
        ########################## DEEP UNCERTAINTIES ######################
        """

        # Equilibrium temperature impact [dC per doubling CO2]/
        # CLimate sensitivity parameter (2.9)
        self.t2xco2 = self.samples_t2xco2[t2xco2_dist][t2xco2_index]
        
        # Choice of the damage function (structural deep uncertainty)
        self.fdamage = fdamage

        # Elasticiy of marginal disutility of damage
        self.emdd = emdd

        """
        ############################# LEVERS ###############################
        """
        # if (self.model_specification == 'EMA_disutility'):
        self.miu = np.zeros((self.steps,))
        # else:
        #     DICE_OPT = pd.read_excel("DICE2013R.xlsm", sheet_name="Opttax",
        #                              index_col=0)
        #     self.miu = np.array(DICE_OPT.iloc[133])
        # Lever: Savings rate
        self.s = np.zeros((self.steps,))

 # Savings rate (optlrsav = 0.2582781457) from the control file
        self.sr = sr
       
        # Initial pure rate of time preference for consumption(0.015)
        self.prtp_con = prtp_con

        # Initial pure rate of time preference for damage(per year) (1)
        self.prtp_dam = prtp_dam

        # Elasticity of marginal utility of consumption
        self.emuc = emuc

        # Upper limit on emissions control rate after 2150 
        self.limmiu = limmiu

        # Initial emissions control rate for base case 2015
        self.miu0 = 0.03
        self.miu[0] = self.miu0
        # if self.model_specification == "EMA_disutility":
        #     self.miu[0] = self.miu0        
        self.miu_period = miu_period

        """
        ##################### ECONOMIC INITIAL VALUES ######################
        """

        # Asymptotic/Maximum world population (in Millions)
        self.max_pop = 10500.0
        # Growth rate to calibrate to 2050 population projection
        self.pop_gr = pop_gr

        # Initial level of total factor productivity (TFP)
        self.tfp[0] = 3.80
        # Initial growth rate for TFP (per 5 years)
        self.tfp_gr[0] = tfp_gr
        # Decline rate of TFP (per 5 years)
        self.decl_tfp_gr = 0.006

        # depreciation rate of capital
        self.dk = 0.100
        # Capital elasticity in production function
        self.gama = 0.300

        # Damage intercept
        self.a1 = 0.0
        # Damage quadratic term
        self.a2 = 0.00267
        # Damage exponent
        self.a3 = 2.00

        # Exponent of control cost function
        self.expcost2 = 2.8
        # Cost of backstop [2005$ per tCO2 2010]
        self.cback = cback
        # Cost decline backstop
        self.decl_back_gr = decl_back_gr

        # Period at which have full participation
        self.periodfullpart = periodfullpart
        # Fraction of emissions under control based on the Paris Agreement
        # US withdrawal would change the value to 0.7086 
        # https://climateanalytics.org/briefings/ratification-tracker/ (0.8875)
        self.partfract2010 = 1

        # raction of emissions under control at full time
        self.partfractfull = 1.0

        # Decline rate of decarbonization (per period)
        self.decl_sigma_gr = -0.001

        # Carbon emissions from land 2010 [GtCO2 per year]
        self.eland0 = 3.3

        # Decline rate of land emissions (per period)
        self.decl_land = 0.2

        # Multiplicitive scaling coefficient
        self.scale1 = 0.016408662
        # Additive scaling coefficient
        self.scale2 = -3855.106895

        # Max cum industrial carbon emissions=Max cum extraction fossil [GtC]
        self.cca_up = fosslim


        """
        ###################### CARBON INITIAL VALUES #######################
        """

        # Initial concentration in atmosphere 2010 [GtC]
        self.mat0 = 830.4
        # Initial concentration in upper strata [GtC]
        self.mu0 = 1527.0
        # Initial concentration in lower strata [GtC]
        self.ml0 = 10010.0
        # Equilibrium concentration in atmosphere [GtC]
        self.mateq = 588.0
        # Equilibrium concentration in upper strata [GtC]
        self.mueq = 1350.0
        # Equilibrium concentration in lower strata [GtC]
        self.mleq = 10000.0

        self.b12 = 0.088
        self.b23 = 0.00250
        self.b11 = 1 - self.b12
        self.b21 = self.b12 * self.mateq / self.mueq
        self.b22 = 1 - self.b21 - self.b23
        self.b32 = self.b23 * self.mueq / self.mleq
        self.b33 = 1 - self.b32

        # 2015 forcings of non-CO2 greenhouse gases (GHG) [Wm-2]
        self.fex0 = 0.25
        # 2100 forcings of non-CO2 GHG [Wm-2]
        self.fex1 = 0.70
        # Forcings of equilibrium CO2 doubling [Wm-2]
        self.fco22x = 3.8

        """
        ###################### CLIMATE INITIAL VALUES ######################
        """

        # Initial lower stratum temperature change [dC from 1900]
        self.tocean0 = 0.0068
        # Initial atmospheric temperature change [dC from 1900]
        self.tatm0 = 0.80
        # Climate equation coefficient for upper level
        self.c1 = 0.098
        # Transfer coefficient upper to lower stratum
        self.c3 = 0.088
        # Transfer coefficient for lower level
        self.c4 = 0.025
        # Climate model parameter
        self.lam = self.fco22x / self.t2xco2

        """
        ####################### LIMITS OF THE MODEL ########################
        """

        # Output low (constraints of the model)
        self.y_lo = 0.000001
        self.ygross_lo = 0.000001
        self.i_lo = 0.000001
        self.c_lo = 2.0
        self.cpc_lo = 0.000001
        self.damages_lo = 2.0
        self.dpc_lo = 0.000001
        self.sdr_con_lo = 0.0
        self.sdr_dam_lo = 0.0
        self.k_lo = 1.0
        # self.miu_up[0] = 1.0

        self.mat_lo = 10.0
        self.mu_lo = 100.0
        self.ml_lo = 1000.0
        self.temp_ocean_up = 20.0
        self.temp_ocean_lo = -1.0
        self.temp_atm_lo = 0.0

        # self.temp_atm_up = 20 or 12 for 2016 version
        self.temp_atm_up = 40.0

        """
        ####################################################################
        #################### MODEL - INITIALISATION ########################
        ####################################################################
        """
        """
        ################# CARBON PARAMETER INTITIALISATION #################
        """

        # Carbon pools
        self.mat[0] = self.mat0
        if(self.mat[0] < self.mat_lo):
            self.mat[0] = self.mat_lo

        self.mu[0] = self.mu0
        if(self.mu[0] < self.mu_lo):
            self.mu[0] = self.mu_lo

        self.ml[0] = self.ml0
        if(self.ml[0] < self.ml_lo):
            self.ml[0] = self.ml_lo

        # Radiative forcing
        self.forcoth[0] = self.fex0
        self.forc[0] = (self.fco22x*(np.log(self.mat[0]/588.000)/np.log(2.0))
                        + self.forcoth[0])

        """
        ################# CLIMATE PARAMETER INTITIALISATION ################
        """

        # Atmospheric temperature
        self.temp_atm[0] = 0.80
        if(self.temp_atm[0] < self.temp_atm_lo):
            self.temp_atm[0] = self.temp_atm_lo
        if(self.temp_atm[0] > self.temp_atm_up):
            self.temp_atm[0] = self.temp_atm_up

        # Oceanic temperature
        self.temp_ocean[0] = 0.0068
        if(self.temp_ocean[0] < self.temp_ocean_lo):
            self.temp_ocean[0] = self.temp_ocean_lo
        if(self.temp_ocean[0] > self.temp_ocean_up):
            self.temp_ocean[0] = self.temp_ocean_up

        """
        ################# ECONOMIC PARAMETER INTITIALISATION ###############
        """

        # Initial world population (in Millions [2015])
        self.pop[0] = 6838.0

        # Initial level of total factor productivity (TFP)
        self.tfp[0] = 3.80
        # Initial growth rate for TFP (per 5 years)
        self.tfp_gr[0] = 0.079

        # Initial capital value 2015 [Trillions 2010 US $]
        self.k[0] = 135.0

        # Gross world product: gross abatement and damages
            # Gama: Capital elasticity in production function
        self.ygross[0] = (self.tfp[0]*((self.pop[0]/1000)**(1-self.gama))
                          * (self.k[0]**self.gama))
        if (self.ygross[0] < self.ygross_lo):
            self.ygross[0] = self.ygross_lo

        # Damage Fraction/temp: Temperature/a1: Damage intercept
            # a2: Damage quadratic term/a3: Damage exponent        
        if (self.fdamage == 0):
            self.damfrac[0] = (self.a1*self.temp_atm[0] 
                               + self.a2*(self.temp_atm[0]**self.a3))
        elif self.fdamage == 1:
            self.damfrac[0] = (1-(np.exp(-0.0025*self.temp_atm[0]**2.45)))
            
        elif self.fdamage == 2:
            self.damfrac[0] = (1-1/(1+0.0028388**2+0.0000050703
                                    *(self.temp_atm[0]**6.754)))
        # Net output (Output - damages)
        self.ynet[0] = self.ygross[0]*(1.0-self.damfrac[0])
        
        # Damages
        self.damages[0] = self.ygross[0] * self.damfrac[0]
        if self.damages[0] < self.damages_lo:
            self.damages[0] = self.damages_lo
        # Damage per capita
        self.dpc[0] = self.damages[0] * 1000 / self.pop[0]
        if self.dpc[0] < self.dpc_lo:
            self.dpc[0] = self.dpc_lo

        ## Output-to-Emission
        # Industrial emissions 2010 [GtCO2 per year]
        self.e0 = 33.61
        # Initial world gross output [Trillions 2015 US $]
        self.q0 = 63.69
        # CO2-equivalent-emissions to output ratio
        self.sigma[0] = self.e0/(self.q0 * (1 - self.miu[0]))
        # Initial growth of sigma (per year)
        self.sigma_gr[0] = sigma_gr

        # Backstop price/cback: cost of backstop
            # decl_back_gr: decline of backstop
        self.pbacktime[0] = self.cback
        # Adjusted cost for backstop
        self.cost1[0] = self.pbacktime[0]*self.sigma[0]/self.expcost2/1000

        # Fraction of emissions under control regime
        if self.periodfullpart == 0:
            self.partfract[0] = 1
        else:
            self.partfract[0] = self.partfract2010

        # Abatement costs
        self.abatecost[0] = (self.ygross[0]*self.cost1[0]
                             * (self.miu[0]**self.expcost2)
                             * (self.partfract[0]**(1-self.expcost2)))

        # Carbon price (unused)
        self.cprice[0] = (self.pbacktime[0]
                          * ((self.miu[0]/self.partfract[0])
                             ** (self.expcost2-1)))

        # Industrial emissions
        self.eind[0] = self.sigma[0]*self.ygross[0]*(1.0-self.miu[0])

        # Emissions from deforestation
        self.etree[0] = self.eland0
        # Cumlative emission from land/tree
        self.cumetree[0] = 100

        # Total emissions
        self.e[0] = self.eind[0]+self.etree[0]

        # Cumulative emissions (cca_up - fossil limits)
        self.cca[0] = 90.0
        # if (self.cca[0] > self.cca_up): #shridhar: how will this (initial) value be < 90 when it's just been defined
        #     self.cca[0] = self.cca_up + 1.0

        self.ccatot[0] = self.cca[0] + self.etree[0]

        # Gross world product (income from production output net of abatement and damages)
        self.y[0] = self.ynet[0]-self.abatecost[0]
        # Applying limits
        if (self.y[0] < self.y_lo): 
            self.y[0] = self.y_lo 

        # Investments & Savings
        # if self.model_specification == "EMA_disutility":
        self.s[0] = self.sr
        
        # Investment
        self.i[0] = self.s[0]*self.y[0]
        if (self.i[0] < self.i_lo):
            self.i[0] = self.i_lo

        ## Consumption
        # Per period consumption
        self.c[0] = self.y[0] - self.i[0]
            # limit self.c_lo = 2.0
        # if self.c[0] < self.c_lo:
        #     self.c[0] = self.c_lo
        # consumption per capita
        self.cpc[0] = self.c[0] * 1000 / self.pop[0]
            # limit self.cpc_lo = 0.0001
        if (self.cpc[0] < self.cpc_lo):
            self.cpc[0] = self.cpc_lo

        ############### Welfare ####################
        # U(C) and V(D) with endogenous discounting
        # period contribution to welfare outcome
        ############################################

        ## Utility of consumption ##
        
        # Initial consumption growth rate
        self.con_g[0] = 0.00 

        # Social discount rate of consumption utility
        # self.sdr_con[0] = 0.00001
        # self.consumption_sdf[0] = 1.00
        self.sdr_con[0] = (self.prtp_con + (self.emuc * self.con_g[0]))
        # if self.sdr_con < self.sdr_con_lo:
        #     self.sdr_con = self.sdr_con_lo

        self.consumption_sdf[0] = 1.00
        
        # Absolute period utility
            # https://www.desmos.com/calculator/75baiw84ym
        
        if self.emuc == 1.00:
            self.inst_util_con[0] = np.log(self.cpc[0])
        else:
            self.inst_util_con[0] = (((self.cpc[0])**(1.0 - self.emuc) - 1.0) / (1.0 - self.emuc) - 1.0)
                
        # Discountedperiod utility 
        self.disc_util_con[0] = self.inst_util_con[0] * self.consumption_sdf[0]

        ## Disutility of Damage V(D)

        # Initial Per capita damage growth (change) rate
        self.dam_g[0] = 0.00

        # Endogenous social discount rate for disutility of damage
        self.sdr_dam[0] = self.prtp_dam + (self.emdd * self.dam_g[0])
        # if self.sdr_dam[0] < self.sdr_dam_lo:
        #         self.sdr_dam[0] = self.sdr_dam_lo

        # Social discount factor for disutility of damage 
        self.damage_sdf[0] = 1.00

        # Instantaneous disutility of damage
        if self.emdd == 1.00:
            self.inst_disutil_dam[0] = np.log(self.dpc[0])
        else:
            self.inst_disutil_dam[0] = (((self.dpc[0]) ** (1.0 - self.emdd) - 1.0)/ (1.0 - self.emdd) - 1.0)
            
        # # Discounted disutility of damage
        self.disc_disutil_dam[0] = self.inst_disutil_dam[0] * self.damage_sdf[0]

        # Welfare function
            # shridhar: check the whole scaling thing and verify formula
        self.period_welfare[0] = (self.disc_util_con[0] - self.disc_disutil_dam[0]) * self.pop[0]/ 1000

        self.welfare[0] = ((self.tstep*self.scale1*np.sum(self.period_welfare))
                        + self.scale2)
        
       
        # logging.info(self, "is initialized.")
        

        """
        ####################################################################
        ########################## MODEL - RUN #############################
        ####################################################################
        """

        for t in range(1, self.steps):

            """
            ####################### CARBON SUB-MODEL #######################
            """

            # Carbon concentration increase in atmosphere [GtC from 1750]
            self.mat[t] = ((self.e[t-1]*(5.0/3.666))
                           + self.b11*self.mat[t-1]+self.b21*self.mu[t-1])
            if(self.mat[t] < self.mat_lo):
                self.mat[t] = self.mat_lo

            # Carbon concentration increase in shallow oceans [GtC from 1750]
            self.mu[t] = (self.b12*self.mat[t-1]+self.b22*self.mu[t-1]
                          + self.b32*self.ml[t-1])
            if(self.mu[t] < self.mu_lo):
                self.mu[t] = self.mu_lo

            # Carbon concentration increase in lower oceans [GtC from 1750]
            self.ml[t] = self.b33*self.ml[t-1]+self.b23*self.mu[t-1]
            if(self.ml[t] < self.ml_lo):
                self.ml[t] = self.ml_lo

            ## Radiative forcing
            # Exogenous forcings from other GHG
            if (t < 19):
                self.forcoth[t] = self.fex0+(1.0/18.0)*(self.fex1-self.fex0)*t
            else:
                self.forcoth[t] = self.fex0+(self.fex1-self.fex0)

            # Increase in radiative forcing [Wm-2 from 1900]
            self.forc[t] = (self.fco22x*(np.log(self.mat[t]/588.000)
                                         / np.log(2.0))
                            + self.forcoth[t])

            """
            ####################### CLIMATE SUB-MODEL ######################
            """

            self.temp_atm[t] = (self.temp_atm[t-1]+self.c1
                                * ((self.forc[t]-((self.fco22x/self.t2xco2)
                                                  * self.temp_atm[t-1]))
                                   - (self.c3*(self.temp_atm[t-1]
                                               - self.temp_ocean[t-1]))))
            if (self.temp_atm[t] < self.temp_atm_lo):
                self.temp_atm[t] = self.temp_atm_lo

            if (self.temp_atm[t] > self.temp_atm_up):
                self.temp_atm[t] = self.temp_atm_up

            self.temp_ocean[t] = (self.temp_ocean[t-1]+self.c4
                                  * (self.temp_atm[t-1]-self.temp_ocean[t-1]))
            if (self.temp_ocean[t] < self.temp_ocean_lo):
                self.temp_ocean[t] = self.temp_ocean_lo

            if (self.temp_ocean[t] > self.temp_ocean_up):
                self.temp_ocean[t] = self.temp_ocean_up

            """
            ###################### ECONONOMIC SUB-MODEL ####################
            """

            # Population and Labour
            self.pop[t] = (self.pop[t-1]
            * (self.max_pop/self.pop[t-1])**self.pop_gr)

            ## Total Factor Productivity
            # TFP growth rate
            self.tfp_gr[t] = self.tfp_gr[0]*np.exp(-1*self.decl_tfp_gr*5*t)
            # Period TFP 
            self.tfp[t] = self.tfp[t-1]/(1-self.tfp_gr[t-1])

            ## Gross Production Output
            
            # k: Captial Stock
            self.k[t] = (((1-self.dk)**self.tstep)*self.k[t-1]
                         + self.tstep*self.i[t-1])
            if (self.k[t] < self.k_lo):
                self.k[t] = self.k_lo

            # Gross world product: Basic C-D production output
                # gama: Capital elasticity in production function
            self.ygross[t] = (self.tfp[t]*((self.pop[t]/1000)**(1-self.gama))
                              * (self.k[t]**self.gama))
            if (self.ygross[t] < self.ygross_lo):
                self.ygross[t] = self.ygross_lo

            # Damage Fraction/temp: Temperature/a1: Damage intercept
                # a2: Damage quadratic term/a3: Damage exponent
            
            if self.fdamage == 0:
                self.damfrac[t] = (self.a1*self.temp_atm[t] 
                                   + self.a2*(self.temp_atm[t]**self.a3))
            elif self.fdamage == 1:
                self.damfrac[t] = (1-(np.exp(-0.0025*self.temp_atm[t]**2.45)))

            elif self.fdamage == 2:
                self.damfrac[t] = (1-1/(1+0.0028388**2+0.0000050703
                                        *(self.temp_atm[t]**6.754)))
            # Net output (Output - damages)
            self.ynet[t] = self.ygross[t]*(1.0-self.damfrac[t])
            # Damages
            self.damages[t] = self.ygross[t]*self.damfrac[t]
            if (self.damages[t] < self.damages_lo):
                self.damages[t] = self.damages_lo

            # Damages per capita
            self.dpc[t] = self.damages[t]*1000/self.pop[t]
            if (self.dpc[t] < self.dpc_lo):
                self.dpc[t] = self.dpc_lo

            # CO2-equivalent-emissions to output ratio
            self.sigma[t] = (self.sigma[t-1]
                             * np.exp(self.sigma_gr[t-1]*self.tstep))
                # Change in sigma: the cumulative improvement in energy efficiency)
            self.sigma_gr[t] = (self.sigma_gr[t-1]
                                * ((1+self.decl_sigma_gr)**self.tstep))
            
            ## Backstop price/cback: cost of backstop
            # decl_back_gr: decline of backstop
            self.pbacktime[t] = self.cback*((1 - self.decl_back_gr)**(t))
            # Adjusted cost for backstop
            self.cost1[t] = self.pbacktime[t]*self.sigma[t]/self.expcost2/1000

            # Fraction of emissions under control regime
            if (t >= self.periodfullpart):
                self.partfract[t] = self.partfractfull
            else:
                self.partfract[t] = (self.partfract2010
                                     + (self.partfractfull-self.partfract2010)
                                     * (t/self.periodfullpart))

            # Emission Control rate

            if self.model_specification == 'EMA_disutility':
                if t >= self.miu_period:
                    self.miu[t] = self.limmiu
                else:
                    self.miu[t] = self.limmiu * t/self.miu_period + self.miu[0]
            

            
            # Abatement costs
            self.abatecost[t] = (self.ygross[t]*self.cost1[t]
                                 * (self.miu[t]**self.expcost2)
                                 * (self.partfract[t]**(1-self.expcost2)))

            # Carbon price (unused)
            self.cprice[t] = (self.pbacktime[t]
                              * ((self.miu[t]/self.partfract[t])
                                 ** (self.expcost2-1)))

            # Industrial emissions
            self.eind[t] = self.sigma[t]*self.ygross[t]*(1.0-self.miu[t])

            # Emissions from deforestation
            self.etree[t] = self.eland0*((1-self.decl_land) ** t)
            
            # Cumulative missions from land
            self.cumetree[t] = self.cumetree[t] + self.etree[t]*(5.0/3.666)

            # Total emissions
            self.e[t] = self.eind[t] + self.etree[t]

            # Cumulative emissions from industry(?) (cca_up - fossil limits)
            self.cca[t] = self.cca[t-1] + self.eind[t-1]*5.0/3.666
            if (self.cca[t] > self.cca_up):
                self.cca[t] = self.cca_up + 1.0

            self.ccatot[t] = self.cca[t] + self.cumetree[t]

            # Gross world product (income from production output net of abatement and damages)
            self.y[t] = self.ynet[t]-self.abatecost[t]       
            # Applying Limits 
               # self.y_lo  = 0.0
            if (self.y[t] < self.y_lo):
                self.y[t] = self.y_lo

            ## Investments & Savings
            # Savings
            self.s[t] = self.sr

            # Investment
            self.i[t] = self.s[t]*self.y[t]
            if (self.i[t] < self.i_lo):
                self.i[t] = self.i_lo

            ## Consumption
            # Period consumption
            self.c[t] = self.y[t] - self.i[t]
                # limit self.c_lo = 2.0
            if (self.c[t] < self.c_lo):
                self.c[t] = self.c_lo 
            # Consumption per capita
            self.cpc[t] = self.c[t]*1000/self.pop[t]
                # limit self.cpc_lo = 0.0001
            if (self.cpc[t] < self.cpc_lo):
                self.cpc[t] = self.cpc_lo 

            ############### Welfare ####################
            # U(C) and V(D) with endogenous discounting
            # period contribution to welfare outcome
            ############################################
            
            ## Utility of comsumption U(C)
            # Per capita consumption growth rate
            self.con_g[t] = (self.cpc[t] - self.cpc[t-1])/ self.cpc[t-1]
            
            # Endogenous social discount rate for utility of consumption
            self.sdr_con[t] = (self.prtp_con + (self.emuc * self.con_g[t]))
            if (self.sdr_con[t] < self.sdr_con_lo):
                self.sdr_con[t] = self.sdr_con_lo
            
            # Social discount Factor for Utility of Consumption 
            self.consumption_sdf[t] = (1.0 /((1.0 + self.sdr_con[t]))**(self.tstep * (t)))
            
            # Absolute period utility
            if (self.emuc == 1.00):
                self.inst_util_con[t] = np.log(self.cpc[t])
            else:
                self.inst_util_con[t] = ((self.cpc[t] ** (1.0 - self.emuc) - 1.0)/ (1.0 - self.emuc) - 1.0)
            
            # Discounted period utility
            self.disc_util_con[t] = self.inst_util_con[t] * self.consumption_sdf[t]
            # debug/ line break
            
            ## Disutility of Damage V(D)
            # Per capita damage growth (change) rate
            self.dam_g[t] = (self.dpc[t] - self.dpc[t-1])/ self.dpc[t-1]
            # self.dam_g[t] =  np.log((self.dpc[t])/(self.dpc[t-1]))
    
            # Endogenous social discount rate for disutility of damage
            self.sdr_dam[t] = (self.prtp_dam + (self.emdd * self.dam_g[t]))
            if (self.sdr_dam[t] < self.sdr_dam_lo):
                self.sdr_dam[t] = self.sdr_dam_lo
            
            # Social discount factor for disutility of damage            
            self.damage_sdf[t] = (1.0 /(1.0 + self.sdr_dam[t]))**(self.tstep * (t))
                      
            # Absolute period disutility
            if (self.emdd == 1.00):
                self.inst_disutil_dam[t] = np.log(self.dpc[t])
            else:
                self.inst_disutil_dam[t] = ((self.dpc[t] ** (1.0 - self.emdd) - 1.0)/ (1.0 - self.emdd) - 1.0)
            
            # Discounted period disutility
            self.disc_disutil_dam[t] = self.inst_disutil_dam[t] * self.damage_sdf[t]

            # Period Welfare term
            self.period_welfare[t] =  (self.disc_util_con[t] - self.disc_disutil_dam[t]) * self.pop[t]/1000

            self.welfare[t] = ((self.tstep*self.scale1*np.sum(self.period_welfare)) + self.scale2)
            """
            ################# POST OPTIMISATION PARAMETERS #################
            """
            ## Endogenous dynamic SCC

            # self.scc[t] = -1000*self.e[t]/(.00001+self.c[t])
            # self.atfrac[t] = ((self.mat[t]-588)/(self.ccatot[t]+.000001))
            # self.atfrac2010[t] = ((self.mat[t]-self.mat[0])
            #                      / (.00001+self.ccatot[t]-self.ccatot[0]))
            # self.ppm[t] = self.mat[t]/2.13

        """
        ####################################################################
        ###################### OUTCOME OF INTEREST #########################
        ####################################################################
        """

        self.data = {'Atmospheric Temperature': self.temp_atm,
                     'Per Capita Damage': self.dpc,
                     'Per Capita Consumption': self.cpc,
                     'Population': self.pop,
                     'Utility of Consumption': self.disc_util_con,
                     'Disutility of Damage': self.disc_disutil_dam,
                     'Welfare': self.welfare,
                     'Total Output': self.y,
                     'Consumption Growth': self.con_g,
                     'Damage Growth': self.dam_g,
                     'Consumption SDR': self.sdr_con,
                     'Damage SDR': self.sdr_dam,
                     
                    }

        return self.data
# %%