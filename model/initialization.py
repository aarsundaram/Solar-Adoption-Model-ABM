import networkx as nx
import random
import numpy as np
import pandas as pd
#import geopandas as gpd
import os
import glob 
import gower 
import dill
from pathos.multiprocessing import ProcessingPool

path = os.getcwd()

rootpath = '/Users/rtseinstein/Documents/GitHub/Solar-Adoption-Agent-based-Model/'

survey = pd.read_csv(rootpath+'data/survey/gps_final_scored_phase3.csv')
survey['PEOPLE_TOT_3PLUS']=survey['PEOPLE_TOT_3PLUS'].fillna(0)
#survey.info()

#read in the main households data
households = pd.read_csv(rootpath+'data/households_main/households_main.csv')
households = households.drop(columns='Unnamed: 0')

demographics = survey[['CASE_ID','AGE_BINNED','INCOME_BINNED','PEOPLE_TOT_3PLUS']].set_index('CASE_ID')
tpb_attributes = survey[['CASE_ID','attitude','subnorms','pbc']].set_index('CASE_ID')

## need to map the numbers to categories in the survey datatable
income_map= {1:'less75k',2:'less75k',3:'75to100k',4:'100to150k',5:'150kplus'}
age_map = {1.:'25to44',2.:'45to54',3.:'55to64',4.:'65plus'}
#need to do this mapping in the households data
size_map={'1person':0.,'2person':0.,'3person':np.random.choice([0.,1.]),'4person':1.,'5person':1.,'6person':1.,'7plus':1.}

demographics['INCOME_BINNED']=demographics['INCOME_BINNED'].map(income_map)
demographics['AGE_BINNED']=demographics['AGE_BINNED'].map(age_map)

households = households.dropna()
households.drop(households.tail(1).index,inplace=True) # drop last n rows

## add the required household row as last row in the demographics survey data
demographics = demographics.rename(columns={'AGE_BINNED':'age','INCOME_BINNED':'income','PEOPLE_TOT_3PLUS':'household_'})

household_demographics= households[['case_id','age','income','household_']]
new_caseids=[]
for i in list(household_demographics['case_id']):
    new_caseids.append(str(i).zfill(9))

household_demographics['case_id']= new_caseids
household_demographics['household_']=household_demographics['household_'].astype('object')
household_demographics= household_demographics.rename(columns={'case_id':'CASE_ID'})
household_demographics = household_demographics.set_index('CASE_ID')


filenames= glob.glob(rootpath+'data/initialization_subsets_24/*.csv')

#definition of initialize function 
def initialize(file):
    df = pd.read_csv(file)
    df_name = file[100:-4]
    print(f'beginning initialization of {df_name}')
    attitude={}
    subnorms={}
    pbc={}
    errorids = []
    for household_case in list(df.index):
        new_record = df.loc[household_case]

        #add to last row of demographics data
        demo = demographics
        demo['household_'] = demo['household_'].astype('object')
        demo = demo.append(new_record)

        #calculate gower's distance
        new_row = gower.gower_matrix(demo)[-1]
        #new_row

        indexes = list(demo.index)

        temp = min(new_row)
        res = [i for i, j in enumerate(new_row) if j == temp]
        #res

        caseids = []
        for i in res:
            caseids.append(indexes[i])


        c = []
        for i in caseids:
            c.append(int(i))

        atts = []
        sn = []
        p = []
        try:
            for i in c:
                if i!=c[-1]:
                    atts.append(tpb_attributes.loc[i]['attitude'])
                    sn.append(tpb_attributes.loc[i]['subnorms'])
                    p.append(tpb_attributes.loc[i]['pbc'])
            
            attitude[household_case]= np.random.choice(atts)
            subnorms[household_case]=np.random.choice(sn)
            pbc[household_case]=np.random.choice(p)
        except:
            attitude[household_case]= 0
            subnorms[household_case]=0
            pbc[household_case]=0
            print(household_case)
            errorids.append(household_case)
            continue
    
    df['attitude']= list(attitude.values())
    df['subnorms']= list(subnorms.values())
    df['pbc']= list(pbc.values())

    df.to_csv(rootpath+f'data/initialization_subsets_24/{df_name}_initialized.csv')
    print(f'finished exporting the initialized file of {df_name}.csv')

pool = ProcessingPool()
results = pool.map(initialize,filenames)


