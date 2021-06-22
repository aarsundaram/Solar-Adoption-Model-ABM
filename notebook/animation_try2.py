import numpy as np
import networkx as nx
import pandas as pd 
import matplotlib.pyplot as plt
import random
from matplotlib import animation


def add_and_remove_edges(G, p_new_connection, p_remove_connection):    
    '''    
    for each node,    
      add a new connection to random other node, with prob p_new_connection,    
      remove a connection, with prob p_remove_connection    

    operates on G in-place    
    '''                
    new_edges = []    
    rem_edges = [] 
    for node in G.nodes():    
        # find the other nodes this one is connected to    
        connected = [to for (fr, to) in G.edges(node)]    
        # and find the remainder of nodes, which are candidates for new edges   
        unconnected = [n for n in G.nodes() if not n in connected]    

        # probabilistically add a random edge    
        if len(unconnected): # only try if new edge is possible    
            if random.random() < p_new_connection:    
                new = random.choice(unconnected)    
                G.add_edge(node, new)    
                #print("\tnew edge:\t {} -- {}".format(node, new)    
                new_edges.append( (node, new) )    
                # book-keeping, in case both add and remove done in same cycle  
                unconnected.remove(new)    
                connected.append(new)    

        # probabilistically remove a random edge    
        if len(connected): # only try if an edge exists to remove    
            if random.random() < p_remove_connection:    
                remove = random.choice(connected)    
                G.remove_edge(node, remove)    
                #print "\tedge removed:\t {} -- {}".format(node, remove)    
                rem_edges.append( (node, remove) )    
                # book-keeping, in case lists are important later?    
                connected.remove(remove)    
                unconnected.append(remove)    
    return rem_edges, new_edges



rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\' 
    
df = pd.read_csv(rootpath+'data\\subset_initialized_latlonvalues.csv')
df = df.drop(columns='Unnamed: 0')
households_in_block = {}
household_ids_in_block = {}
graph_dict = {}   
p_new_connection=0.85
p_remove_connection= 0.85                                             # holds all the graphs indexed by blockid [geoid]
G = nx.Graph()

#now i need to get number of geoids unique 
for block in df['geoid'].unique():  
    G_temp=nx.Graph()
    households_in_block[block] = df[df['geoid']==block]                 # contains all the information about the households 
    household_ids_in_block[block] =  df[df['geoid']==block]['CASE_ID'].values  
                                                                        # contains only their ID
                                                                        # you only need id to initialize a node
    tempdf = households_in_block[block]
    for household in household_ids_in_block[block]:
        lon = tempdf.loc[tempdf['CASE_ID']==household,'lon'].values[0]
        lat = tempdf.loc[tempdf['CASE_ID']==household,'lat'].values[0]        
        
        G_temp.add_node(household, pos=(lon,lat))
        G.add_node(household, pos=(lon,lat))
        ## add G to the dictionary
        graph_dict[block] = G_temp



main_graph_dict={}

#rem_edges, new_edges = add_and_remove_edges(G, p_new_connection, p_remove_connection)
#G.remove_edges_from(rem_edges)
#G.add_edges_from(new_edges)

for i in range(5):
    main_graph_copy = G

    for block in list(graph_dict.keys()):
        tempG = graph_dict[block]  #gets the graph of that block 
                                #draw random edges between pairs of them
        rem_edges, new_edges = add_and_remove_edges(tempG, p_new_connection, p_remove_connection)
        tempG.remove_edges_from(rem_edges)
        tempG.add_edges_from(new_edges)
        main_graph_copy= nx.compose(main_graph_copy,tempG)

    main_graph_dict[i] = main_graph_copy

print(main_graph_dict)