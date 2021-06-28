from mesa import Agent
from enum import Enum
import math 
import random 
from mesa.space import NetworkGrid
from networkx.classes.function import selfloop_edges


# ---------------------------------------------------------------
class Household(Agent):
 

    def __init__(self, unique_id, model, income,age,size,ami_category,elec_consumption,attitude,attitude_uncertainty, pbc,subnorms,geoid,tract,bgid,ToleratedPayBackPeriod,circle1, circle2, circle3, geolinks,adoption_status=0):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.income = income
        self.age = age
        self.size = size
        self.ami_category = ami_category
        self.elec_consumption = elec_consumption
        self.attitude= attitude
        self.attitude_uncertainty = attitude_uncertainty
        self.pbc = pbc
        self.subnorms = subnorms
        self.geoid = geoid
        self.tract=  tract
        self.bgid = bgid
        self.intention = self.model.tpb_constant+ (self.model.pbc_weight * self.pbc) + (self.model.att_weight* self.attitude) + (self.model.sn_weight*self.subnorms)
        self.orientation = random.choice([1,2,3,4])  #orientiation of the solar panel on their roof to calculate the azimuth angle 
                    # 1= north
                    # 2 = east
                    # 3 = west
                    # 4 = south 
        self.ToleratedPayBackPeriod = ToleratedPayBackPeriod
        self.circle1 = circle1
        self.circle2 = circle2        
        self.circle3 = circle3     
        self.geolinks = geolinks
        
        self.adoption_status = adoption_status
   
        self.SimplePayBackPeriod = 9  #by default
    
    def timestep_to_year(self):
        timestep = self.model.schedule.steps
        for i in range(self.model.run_time + 1):
            if timestep in range((i*12)-12+1, (i*12)+1):
                year = 2015 + (i-1) + 1
        return year


    def pbc_evolution(self):
        """
        Models the evolution of a household's perceived behavioral control over the action of
        investing in a solar panel 
        """
        timestep_to_year = {0:2015,1:2015,2:2016,3:2017,4:2018,5:2019,6:2020,7:2021,8:2022,9:2023,10:2024,11:2025,12:2026}
        #values that have to be changed to make dynamic 
        TimeStepYear = timestep_to_year[self.model.schedule.steps]
        #south facing: 
        annualsolarproduction_dict = {1:1238,2:2475,3:3714,4:4954,5:6189,6:7427,7:8665,8:9904,9:10904,10:11004,11:12004,12:13004}
        AvgPricePerWattSolar_dict = {2015:3.9,                 #2015
                                     2016:3.6,                 #2016
                                         2017:3.3,             #2017
                                             2018:3.1,         #2018
                                               2019:3.1,
                                               2020: 3.1,
                                               2021: 2.73}       #2019
        RetailElectricityRate_dict = {2015: 0.1854, 
                                    2016: 0.1758,
                                    2017: 0.1803,
                                    2018: 0.1852,
                                    2019: 0.1794,
                                    2020: 0.1717,
                                    2021: 0.2190
                                    }

        AvgPricePerWattSolar = AvgPricePerWattSolar_dict[TimeStepYear] 
        RetailElectricityRate = RetailElectricityRate_dict[TimeStepYear]
        #FederalTaxCredit = 0.55  # new york tax 25% of cost + federal tax credits: 26% (see document) 
        #FederalTaxCredit = 0.46  
        #FederalTaxCredit = 0.51
        #FederalTaxCredit = 0.56

        #Scenario-2 : Income based tax-credit
        FederalTaxCredit= {'less75k':0.56, '75to100k':0.51,'100to150k':0.46,'150kplus':0.46}
        ProductionRatioOfPanel = 1.03 
        UtilityRebate = 1000 # dollars
        
        ### Formulae
                                    ## elec_consumption is in $/month. divide by 0.19 to convert to KWh/month
        AnnualElecConsumption = (( self.elec_consumption / 0.19 ) * 12 )    #in KWh/year 

        SystemSize = (AnnualElecConsumption / ProductionRatioOfPanel)  # production rate of panels in NY = 1.23 
                                                                        # gives SystemSize in Watts 

        NetPanelCost = ((SystemSize * AvgPricePerWattSolar) - UtilityRebate) * (FederalTaxCredit[self.income]) 

        AnnualSolarProduction = annualsolarproduction_dict[int(SystemSize/1000)+4]   # TODO : info in solar_prices excel sheet in solar_potential

        AnnualSavings=  (AnnualSolarProduction*RetailElectricityRate) -(AnnualElecConsumption* RetailElectricityRate) 

        self.SimplePayBackPeriod = int(NetPanelCost/AnnualSavings)

        
        if self.SimplePayBackPeriod <  self.ToleratedPayBackPeriod:
            if math.floor(abs(self.ToleratedPayBackPeriod-self.SimplePayBackPeriod)) in range(0,2):
                latest_pbc = self.model.pbc_dict[self][self.model.schedule.steps][-1]
                self.model.pbc_dict[self][self.model.schedule.steps].append(min(latest_pbc+0.5, 1))
            #if math.floor(abs(self.ToleratedPayBackPeriod-self.SimplePayBackPeriod)) in range(1,4):
            #    self.pbc= min(self.pbc+0.15,1)            
            if  (self.ToleratedPayBackPeriod-self.SimplePayBackPeriod) >=1:
                self.model.pbc_dict[self][self.model.schedule.steps].append(1)
 
        
        #print('At timestep',self.model.schedule.steps,'agent',self.unique_id,'has simple payback of',self.SimplePayBackPeriod,'and a tolerated payback of',self.ToleratedPayBackPeriod)
        #print('new pbc is of agent ', self.unique_id,'is', self.pbc)



    def step(self):
        if self.adoption_status==0:
            self.pbc_evolution()


    def __str__(self):
        return type(self).__name__ + str(self.unique_id)
