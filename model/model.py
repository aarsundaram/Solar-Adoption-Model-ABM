from enum import unique
from mesa import Model, model
from mesa.time import BaseScheduler
from mesa.space import NetworkGrid
import random
#from mesa.visualization.modules import NetworkModule
from households import Household
import pandas as pd
from collections import defaultdict
import networkx as nx

# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max


# ---------------------------------------------------------------
class AdoptionModel(Model):
    
    step_time = 1

    # file_name = '../data/demo-4.csv'
    rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\'

    def __init__(self, seed=None, x_max=500, y_max=500, x_min=0, y_min=0):
        
        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.space = None
        #self.sources = []
        #self.sinks = []
        self.G = nx.Graph()
        self.graph_dict = {}

        self.generate_model()

    def generate_model(self):
        """
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """
        rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\' 
        
        df = pd.read_csv(rootpath+'data\\subset_initialized_latlonvalues.csv')
        df = df.drop(columns='Unnamed: 0')
        households_in_block = {}
        household_ids_in_block = {}
                                                                 # holds all the graphs indexed by blockid [geoid]
        
        def add_and_remove_edges(G, p_new_connection, p_remove_connection):    

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
                
                G_temp.add_node(str(household), pos=(lon,lat))
                self.G.add_node(str(household), pos=(lon,lat))
                    
            ## add G to the dictionary
            self.graph_dict[block] = G_temp
        
        
        rem_edges, new_edges = add_and_remove_edges(self.G, 0.5, 0.5)
        self.G.remove_edges_from(rem_edges)
        self.G.add_edges_from(new_edges)

        

        self.grid= NetworkGrid(self.G)
        
        for _, row in df.iterrows():  # index, row in ...
  
            agent = Household(unique_id = str(row['CASE_ID']),
                             model = self, 
                             income = row['income'],
                             age= row['age'],
                             size= row['household_'],
                             ami_category = row['ami_categ'],
                             elec_consumption= row['elec_consumption'],
                             attitude = row['attitude'],
                             pbc = row['pbc'],
                             subnorms = row['subnorms'],
                             geoid = row['geoid'],
                             tract = row['tract'],
                             bgid = row['bgid'],
                             adoption_status = 0)
            
            

            if agent:
                self.schedule.add(agent)
                y = row['lat']
                x = row['lon']
                self.grid.place_agent(agent, node_id=agent.unique_id)
                #self.space.place_agent(agent, (x, y))
                #agent.pos = (x, y)



    def step(self):
        """
        Advance the simulation by one step.
        """
        #nx.draw_networkx(self.G, nx.get_node_attributes(self.G, 'pos'))
        self.schedule.step()

# EOF -----------------------------------------------------------
