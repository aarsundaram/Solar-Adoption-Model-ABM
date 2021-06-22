import pandas as pd 
import os 
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

plt.close('all')
from matplotlib.animation import FuncAnimation


rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\Solar-Adoption-Agent-Based-Model\\' 
interactions = pd.read_csv(rootpath+'experiment\\interactions_mainsubset_36runs.csv')
interactions = interactions.drop(columns='Unnamed: 0')
interactions['first_agent'] = interactions['first_agent'].astype('string') 
interactions['second_agent'] = interactions['second_agent'].astype('string') 

#now import the latlon values file 
subset = pd.read_csv(rootpath+'data\\households_subset\\subset_initialized_latlonvalues.csv')
subset = subset.drop(columns='Unnamed: 0')
# create map of values 
subset['CASE_ID']=subset['CASE_ID'].astype(int)
subset = subset.set_index('CASE_ID')
subset['lat']= subset['lat'].astype(float)
subset['lat']= subset['lat'].astype(float)

latmap = subset['lat'].to_dict()
lonmap = subset['lon'].to_dict()

graph_dict =  {}
for step in list(interactions['timestep'].unique()):
    #get a dataframe of the timestep
    timestepdf = interactions.loc[interactions['timestep']==step]
    G=nx.MultiGraph()
    for _,row in timestepdf.iterrows():
        # add subgraphs which will then be added to the main graph_dict
        first_agent = int(row['first_agent'])
        G.add_node(first_agent, pos=(latmap[first_agent],lonmap[first_agent])) 
        second_agent = int(row['second_agent'])
        G.add_node(second_agent, pos=(latmap[second_agent],lonmap[second_agent]))

        G.add_edge(first_agent,second_agent)

    graph_dict[step]=G 

 

fig,ax = plt.subplots(figsize=(15,15))

def update(step):
    return nx.draw(graph_dict[step], nx.get_node_attributes(graph_dict[step], 'pos'), ax=ax)

for step in list(graph_dict.keys()):
    a= animation.FuncAnimation(fig, update(step), interval = 2, frames=4, blit=False)
    fig = plt.gcf()

    plt.show()




    
