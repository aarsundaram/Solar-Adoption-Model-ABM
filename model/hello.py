
import pandas as pd 
import networkx as nx 
from enum import unique
from mesa import Model, Agent
from mesa.time import BaseScheduler
from mesa.space import NetworkGrid
import random
import pandas as pd
import networkx as nx
#import os 

#path = os.getcwd()
#rootpath = '/home/nfs/ameenakshisund/abm/Solar-Adoption-Agent-based-Model'
rootpath = '/Users/rtseinstein/Documents/GitHub/Solar-Adoption-Agent-based-Model/'
class myagent(Agent):
   def __init__(self, unique_id, model):
       super().__init__(unique_id, model)

       self.unique_id = unique_id
       self.model = model 

   def step(self):
       print(self.unique_id, 'says Hello!')
       print(self.unique_id,'prints out a random number:',random.choice[1,2])


class MyModel(Model):
    
    def __init__(self):
        self.G = nx.Graph()
        self.schedule = BaseScheduler(self)
        agent = myagent(unique_id='blue',model=self)
        self.schedule.add(agent)
        self.G.add_node(agent,label=agent.unique_id)

        self.datacollector_df = pd.DataFrame()

    def step(self):
        print('Model is created')
        agents = []
        for agent in self.schedule.agents:
            print('One agent called', agent.unique_id,'has been created')
            agents.append(agent.unique_id)

        self.datacollector_df['agent_name']= agents 
        


sample = MyModel()
sample.step()

sample.datacollector_df.to_csv(rootpath+'/experiment/hello_output.csv')

    
