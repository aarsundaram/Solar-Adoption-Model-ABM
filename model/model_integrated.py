from enum import unique
#from matplotlib.animation import TimedAnimation
from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import NetworkGrid
import random
#from networkx.generators.small import house_graph
import pandas as pd
import networkx as nx
import math
#import multiprocessing as mp 
import dill 
from pathos.multiprocessing import ProcessingPool
from households import Household
#import resource 
import sys 
import glob

random.seed(123)

sys.setrecursionlimit(10000)

class AdoptionModel(Model):
    
    
    def __init__(self,filename):
        
        self.filename = filename
        self.schedule = BaseScheduler(self)
        self.space = None
        self.run_time = 8
        self.G = nx.Graph()
        self.seeded_agents = []
        self.social_network= nx.Graph()

        ## Interaction Groups 
        self.geoid_dict = {}                        ## Geographic interaction
        self.bgid_dict= {}                          ## circle-2 interaction
        self.all_households= []                     ## to hold list of all households in albany
                                                    
        
        ## TPB attributes
        self.tpb_constant = 0.2799
        self.pbc_weight = 0.1409
        self.att_weight = 0.4717
        self.sn_weight = 0.1081  # TODO : substitute these values with ones from regression results : DONE 
        

        #self.intention_threshold = {0:0.80,1:0.80,2:0.95,3:0.95,4:0.95,5:0.95,6:0.95,7:0.95,8:0.95,9:0.95}
        #doesn't work
        #self.intention_threshold = {0:0.80,1:0.80,2:0.81,3:0.83,4:0.87,5:0.90,6:0.92,7:0.92,8:0.93,9:0.94}

        #final trial for calibration
        self.intention_threshold = {0:0.80,1:0.80,2:0.82,3:0.87,4:0.89,5:0.91,6:0.92,7:0.93,8:0.93,9:0.94}

        self.datacollector_df = pd.DataFrame(columns = ['timestep','case_id','attitude','subnorms','pbc','adoption_status','geoid'])
        self.interactions_df = pd.DataFrame(columns=['timestep','first_agent','agent1_initial_attitude','agent1_final_attitude','second_agent','agent2_initial_attitude','agent2_final_attitude'])
        
        ## Dictionaries 
        self.attitude_dict= {}
        self.attitude_uncertainty_dict={} 
        self.pbc_dict={}
        self.subnorms_dict = {}
        
        self.generate_model()

    def generate_model(self):
        """
        - Adds household agents from the file as Mesa Agents 
        - Adds these Mesa Household Agents as node in Graph G 
        - Block-wise households are stored in dictionary form

        """ 
        #rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\Solar-Adoption-Agent-Based-Model\\' 
        rootpath = '/home/nfs/ameenakshisund/abm/Solar-Adoption-Model-ABM/'
        #rootpath= '/Users/rtseinstein/Documents/GitHub/Solar-Adoption-Model-ABM/'
        
        #df = pd.read_csv(rootpath+'data\\households_subset\\subset_initialized_latlonvalues.csv')
        #df = pd.read_csv(rootpath+'data/households_subset/subset_initialized_latlonvalues.csv')
        df = pd.read_csv(self.filename)
        df = df.drop(columns='Unnamed: 0') 
        #df = df.head(3000)

        households_main = pd.read_csv(rootpath+'data/households_main/households_main_initialized.csv')
        households_main = households_main.drop(columns='Unnamed: 0')

        # create an empty dictionary for storing agents neighborhood-wise (geoid) 
        for geoid in list(df['GEOID10'].unique()):
            self.geoid_dict[geoid] =[]


        for blockgroup in list(df['bgid'].unique()):
            self.bgid_dict[blockgroup]=[]

        
        #####################################################
        #### INITIALIZING TOLERATED PAYBACK PERIOD  #########
        #####################################################

        pbc_dict1 = {}
        for _,row in df.iterrows():
            pbc_dict1[row['case_id']] = [row['pbc']]

        values = list(df['pbc'])
        min_value = min(values)
        max_value = max(values)

        for caseid in list(pbc_dict1.keys()):
            fraction = ((pbc_dict1[caseid][0] - min_value) / (max_value - min_value))*100

            pbc_dict1[caseid].append(fraction)
            if 0<= fraction <1:
                pbc_dict1[caseid].append(0)
            if 1<= fraction <5:
                pbc_dict1[caseid].append(3)
            if 5<= fraction <10: 
                pbc_dict1[caseid].append(5)
            if 10<= fraction <25: 
                pbc_dict1[caseid].append(6)
            if 25<= fraction <35: 
                pbc_dict1[caseid].append(7.5)
            if 35<= fraction <40: 
                pbc_dict1[caseid].append(8)
            if 40<= fraction <50: 
                pbc_dict1[caseid].append(10)
            if 50<= fraction <78: 
                pbc_dict1[caseid].append(15)
            if 78<= fraction <95: 
                pbc_dict1[caseid].append(20)
            if 95<= fraction <100: 
                pbc_dict1[caseid].append(25)
            if fraction== 100: 
                pbc_dict1[caseid].append(30)

        tolerated_paybackperiods= []
        for i in list(pbc_dict1.values()):
            tolerated_paybackperiods.append(i[2])
        
        # add this column to df
        df['toleratedpayback']= tolerated_paybackperiods
        #print(tolerated_paybackperiods)

        for _,row in df.iterrows():
            agent = Household(unique_id = str(row['case_id']),
                             model = self, 
                             income = row['income'],
                             age= row['age'],
                             size= row['household_'],
                             ami_category = row['ami_catego'],
                             elec_consumption= row['elec_consu'],
                             attitude = row['attitude'],
                             attitude_uncertainty = 1-abs(row['attitude']),
                             pbc = row['pbc'],
                             subnorms = row['subnorms'],
                             geoid = row['GEOID10'],
                             tract = row['TRACTCE10'],
                             bgid = row['bgid'],
                             ToleratedPayBackPeriod= row['toleratedpayback'],
                             circle1=[],
                             circle2=[],
                             circle3=[],
                             geolinks=[],
                             adoption_status = 0)

            if agent:
                #print('agent',agent.unique_id,'initial_pbc:',agent.pbc,'with tolerated payback of',agent.ToleratedPayBackPeriod)
                
                self.schedule.add(agent)
                self.G.add_node(agent, label=agent.unique_id)   # main graph holds all agent nodes
                

                #preparing dictioanries that will enable interactions 
                self.all_households.append(agent)
                self.geoid_dict[agent.geoid].append(agent)
                self.bgid_dict[agent.bgid].append(agent)
                
                # preparing tpb dictionaries
                self.attitude_dict[agent]={}
                self.attitude_dict[agent][self.schedule.steps]= []
                self.attitude_dict[agent][self.schedule.steps].append(agent.attitude)

                self.attitude_uncertainty_dict[agent]={}
                self.attitude_uncertainty_dict[agent][self.schedule.steps]=[]
                self.attitude_uncertainty_dict[agent][self.schedule.steps].append(agent.attitude_uncertainty)  #update with initialized value for timestep 0
 
                self.pbc_dict[agent]={}
                self.pbc_dict[agent][self.schedule.steps]= []
                self.pbc_dict[agent][self.schedule.steps].append(agent.pbc)  

                self.subnorms_dict[agent]={}
                self.subnorms_dict[agent][self.schedule.steps]= []
                self.subnorms_dict[agent][self.schedule.steps].append(agent.subnorms)        


        # #TODO: Scenario5: random seeding
        # seed_agents = random.choices(self.schedule.agents,k=math.ceil(0.01*len(df)))
        # if len(seed_agents)>0:
        #     for agent in seed_agents:
        #         self.seeded_agents.append(agent.unique_id)
        #         agent.adoption_status = 1

        #TODO: Scenario-03: Seed Individuals by income-group 

        ## 3A : LOW INCOME GROUP SEEDING
        # sample_percentage = 0.02 ## modify this
        # #get agents in the tract who are low income. then sample from 0.1% of them 
        # low_income_agents = [hh for hh in self.schedule.agents if hh.income=='less75k']
        # seed_agents = random.choices(low_income_agents, k=math.ceil(sample_percentage*len(df)))
        # if len(seed_agents)>0:
        #     for agent in seed_agents:
        #         self.seeded_agents.append(agent.unique_id)
        #         agent.adoption_status = 1

        ## 3B: LOW & MIDDLE-INCOME GROUP SEEDING:
        # same sample percentage as above
        sample_percentage = 0.01
        low_middle_income_agents = [hh for hh in self.schedule.agents if hh.income in ['less75k','75to100k']]
        seed_agents = random.choices(low_middle_income_agents, k=math.ceil(sample_percentage*len(df)))
        if len(seed_agents)>0:
            for agent in seed_agents:
                self.seeded_agents.append(agent.unique_id)
                agent.adoption_status = 1 



        #######################################
        ######## CIRCLES OF INFLUENCE #########
        #######################################
    
        # initializing their networks 

        # after all the agents have been initialized, now give them coregroups
        for agent in self.schedule.agents:
            
            #geolinks at geoid level
            #initialize upto 10 neighs that will be in their network
            neighs = random.choices(self.geoid_dict[agent.geoid],k=min(len(self.geoid_dict[agent.geoid]),10))
            neighs = [i for i in neighs if i!=agent]
            for neigh in neighs:
                self.social_network.add_edge(agent,neigh)
            
            #circle2 at bgid level 
            #initialize upto 50 of them who will be in their network.
            # they will actually interact with just 15 of them max every month
            bgid_neighs = random.choices(self.bgid_dict[agent.bgid],k=min(50,len(self.bgid_dict[agent.bgid])))
            bgid_neighs = [i for i in bgid_neighs if i!=agent]

            #add 3 from these permanently to circle1
            circle1 = random.choices(bgid_neighs,k=min(3,len(bgid_neighs)))
            #remove these core members from the bgid_neighs to prevent double interaction
            bgid_neighs = [i for i in bgid_neighs if i not in circle1]
            for bgid_neigh in bgid_neighs:
                self.social_network.add_edge(agent,bgid_neigh)



            #circle3 at albany level
            #initialize upto 200 of them.
            #although at every timestep, an agent interacts only with upto 20 of them
            thirdcircle = random.choices(self.all_households,k=200)
            thirdcircle = [i for i in thirdcircle if i!=agent]
            #adding 2 from these permanently to core group
            circle1 = circle1+ random.choices(thirdcircle, k=min(2,len(thirdcircle)))
            thirdcircle = [i for i in thirdcircle if i not in circle1]

            for i in thirdcircle:
                if i not in list(self.G.nodes()):
                    self.G.add_node(i)
            
            for hh in thirdcircle:
                self.social_network.add_edge(agent,hh)

            for core in circle1:
                self.social_network.add_edge(agent,core) 
                
            # initializing the networks.
            #no edges at this stage. only at the attitude evolution step
            agent.geolinks = neighs
            agent.circle2 = bgid_neighs
            agent.circle3= thirdcircle
            agent.circle1 = circle1

        
        ## TODO: SCENARIO-04 : Seeding Influencers 

        ## 4A: Seeding Influencers randomly (from any income group)
        # sample_number = math.ceil(0.001*len(df))  ##number instead of percentage 
        # print('beginning scenario-04')
        # #using degree centrality as a measure of influence in an integrated network 
        # influencers_anygroup = nx.degree_centrality(self.social_network)
        # influencers_anygroup2 = sorted(influencers_anygroup.items(), key=lambda x:x[1]) ## getting all influencers 
        # influencer_caseids = [influencers_anygroup2[i][0] for i in range(len(influencers_anygroup2))][-sample_number:]
        # print('len of influencer caseids',len(influencer_caseids))
        # for agent in influencer_caseids:
        #     #agent = self.schedule._agents[influencer]
        #     self.seeded_agents.append(agent.unique_id)
        #     agent.adoption_status =1 
        # print('finished seeding')

        
        # ##4B: LOW-INCOME GROUP INFLUENCERS SEEDING
        # sample_number = math.ceil(0.001*len(df))  ##number instead of percentage 
        # influencers_anygroup = nx.degree_centrality(self.social_network)
        # influencers_anygroup2 = sorted(influencers_anygroup.items(), key=lambda x:x[1]) ## getting all influencers
        # ##get caseids 
        # influencer_caseids = [influencers_anygroup2[i][0] for i in range(len(influencers_anygroup2))]
        # influencers_lowincome = []
        # for agent in influencer_caseids:
        #     #agent = self.schedule._agents[influencer]
        #     if agent.income == 'less75k':
        #         influencers_lowincome.append(agent)

        # ##seed 0.1% of the agents (of the sample number size)
        # seed_agents = influencers_lowincome[-sample_number:]
        # if len(seed_agents)>0:
        #     for agent in seed_agents:
        #         self.seeded_agents.append(agent.unique_id)
        #         agent.adoption_status=1

        # ##4C: LOW & MIDDLE INCOME GROUP INFLUENCERS SEEDING
        # sample_number = math.ceil(0.001*len(df))  ##number instead of percentage 
        # influencers_anygroup = nx.degree_centrality(self.social_network)
        # influencers_anygroup2 = sorted(influencers_anygroup.items(), key=lambda x:x[1]) ## getting all influencers
        # ##get caseids 
        # influencer_caseids = [influencers_anygroup2[i][0] for i in range(len(influencers_anygroup2))]
        # influencers_low_middle_income = []
        # for agent in influencer_caseids:
        #     #agent = self.schedule._agents[influencer]
        #     if agent.income in ['less75k','75to100k']:
        #         influencers_low_middle_income.append(agent)

        # ##seed 0.1% of the agents (of the sample number size)
        # seed_agents = influencers_low_middle_income[-sample_number:]
        # if len(seed_agents)>0:
        #     for agent in seed_agents:
        #         self.seeded_agents.append(agent.unique_id)
        #         agent.adoption_status=1       

    def attitude_evolution(self):
        """
        Using the relative aggreement algorithm, model interactions between agents at:
        - block level (physical interactions bound by geography)
        - socioeconomic level (interactions within different circles of the agent)
        """

        #At every timestep, go through every agent, get their 4 circles
        # create random interactions within their social networks 
        # store these interactions in "interactions"
        # return this to the model
        # in a for-loop, the model goes through every interaction and updates opinions
        interactions = []

        def add_and_remove_edges(G, p_new_connection, p_remove_connection):    
            """
            where:
            G : input graph 
            p_new_connection: probability of forming new connection
            p_remove_connection: probability of removing existing connection

            returns: rem_edges, new_edges
            rem_edges : list of edges removed
            new_edges : list of edges formed 

            """ 

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


        def ra_implementation(a,b, mu=0.2):
            """
            where:
            a = first mesa agent in the interactions pair
            b = second mesa agent in the interactions pair
            
            returns: attitude (float64) of first agent, attitude of second agent

            """
            ## DONE : add the uncertainty values from a V-shaped map that is derived from the 
            ## agent's attitude itself. If extreme, more certain. Centrist: more uncertain.

            if a.adoption_status==1:
                mu = 0.5  #intensity of interactions is more if a is an adopter
            else:
                mu=0.2
                        # TODO: sensitivity analysis on this variable 
            #print('a:',a.unique_id,'b',b.unique_id)

            h_ij=0
            #x_i= a.attitude
            x_i = self.attitude_dict[a][self.schedule.steps][-1]
            #print('initial x_i', x_i)
            #u_i= a.attitude_uncertainty   ## DONE: this also has to be initialized at the very beginning for this to change. 
            u_i = self.attitude_uncertainty_dict[a][self.schedule.steps][-1]

            #x_j= b.attitude
            x_j = self.attitude_dict[b][self.schedule.steps][-1]
            #print('initial x_j', x_j)
            #u_j= b.attitude_uncertainty
            u_j =  self.attitude_uncertainty_dict[b][self.schedule.steps][-1] 



            h_ij=min(x_i+u_i, x_j+u_j) - max(x_i-u_i, x_j-u_j)

            if(h_ij>u_i):
                
                relagree=(h_ij/u_i)-1
                #print('relagree', relagree)
                delta_x_j=mu*relagree*(x_i-x_j)
                #print('delta_x_j:', delta_x_j)
                delta_u_j=mu*relagree*(u_i-u_j)
                #print('delta_u_j',delta_u_j)
                x_j=x_j+delta_x_j
                #print('new x_j:', x_j)
                u_j=u_j+delta_u_j
                #print('new x_i', x_i)
                
                #print('x_i,  x_j =', x_i, x_j )
                #print('timestep:',self.schedule.steps,'\n')
                #print("influence! dx, du=",delta_x_j,delta_u_j)

            #opinions change only for non-adopters 
                #updating uncertainty of b
                self.attitude_uncertainty_dict[b][self.schedule.steps].append(u_j)

            return x_i,x_j


        def circles_of_influence(interaction,mu=0.2):
            
            ## making it directional on adopters # TODO: just marking for easy access. 
            if (interaction[0].adoption_status ==1) and (interaction[1].adoption_status==1): #if both agents are adopters, no need RA-implementation

                self.attitude_dict[interaction[0]][self.schedule.steps].append(self.attitude_dict[interaction[0]][self.schedule.steps][-1])   #value remains the same basically, no change
                self.attitude_dict[interaction[1]][self.schedule.steps].append(self.attitude_dict[interaction[1]][self.schedule.steps][-1])   #value remains the same basically, no change

                print('Time step:', self.schedule.steps,',both', interaction[0].unique_id,'and',interaction[1].unique_id,'are adopters, no change in opinions!')


            if (interaction[0].adoption_status==1) and (interaction[1].adoption_status==0): 
                first_agent = interaction[0]
                second_agent = interaction[1]
                
                first_agent_initial_attitude = self.attitude_dict[first_agent][self.schedule.steps][-1]
                second_agent_initial_attitude= self.attitude_dict[second_agent][self.schedule.steps][-1]

                first_agent_new_attitude, second_agent_new_attitude = ra_implementation(first_agent,second_agent, mu)
                self.attitude_dict[first_agent][self.schedule.steps].append(first_agent_new_attitude)
                self.attitude_dict[second_agent][self.schedule.steps].append(second_agent_new_attitude) 
                
            else:
                first_agent = interaction[1]
                second_agent= interaction[0]

                first_agent_initial_attitude = self.attitude_dict[first_agent][self.schedule.steps][-1]
                second_agent_initial_attitude= self.attitude_dict[second_agent][self.schedule.steps][-1]

                first_agent_new_attitude, second_agent_new_attitude = ra_implementation(first_agent,second_agent, mu)
                self.attitude_dict[first_agent][self.schedule.steps].append(first_agent_new_attitude)
                self.attitude_dict[second_agent][self.schedule.steps].append(second_agent_new_attitude) 

            
        for agent in self.schedule.agents:
            tempG = nx.Graph()
            for i in random.choices(agent.geolinks,k=min(len(agent.geolinks),5)):
                tempG.add_edge(agent,i)
            for edge in list(tempG.edges()):
                circles_of_influence(edge,mu=0.2)

            tempG = nx.Graph()
            for i in agent.circle1:
                tempG.add_edge(agent,i)
            for edge in list(tempG.edges()):
                circles_of_influence(edge,mu=0.5)

            tempG = nx.Graph()
            for i in random.choices(agent.circle2, k=min(15,len(agent.circle2))):
                tempG.add_edge(agent,i)
            for edge in list(tempG.edges()):
                circles_of_influence(edge, mu=0.1)

            tempG = nx.Graph()
            for i in random.choices(agent.circle3,k=min(20,len(agent.circle3))):
                tempG.add_edge(agent,i)
            for edge in list(tempG.edges()):
                circles_of_influence(edge,mu=0.05)


    def subnorms_evolution(self):
        """
        Modelling the influence of adopters in the block, on adoption decisions of other block residents

        """ 
        for household in self.schedule.agents:
            # get the number of adopters in the household's block
            adopters_in_block = []
            for hh in self.geoid_dict[household.geoid]:
                if hh.adoption_status==1:
                    adopters_in_block.append(hh)
                    self.subnorms_dict[hh][self.schedule.steps].append(1)

            if household.adoption_status==0:
                if len(adopters_in_block) > (0.33 * len(self.geoid_dict[household.geoid])): 
                    self.subnorms_dict[household][self.schedule.steps].append(min(household.subnorms+0.5,1))       
                                            ## if more than one-third of the neighbors have a solar panel on their roof, 
                                            ## the household's subnorms becomes increases by 0.5 
                                            ## any scientific explanation for the threshold? 
                if len(adopters_in_block) > (0.5 * len(self.geoid_dict[household.geoid])): 
                    self.subnorms_dict[household][self.schedule.steps].append(1)                
                                            ## if more than 1/2 of the neighbors have a solar panel on their roof, 
                                            ## the household's subnorms becomes increases to 1 
                                            ## any scientific explanation for the threshold? 
        

    def step(self):

        """
        Advance the model by a step
        """
        self.schedule.step()  ## this is what increments step by 1, else it prints previous step 

        #print(self.schedule.steps)
        print('Timestep:', self.schedule.steps)

        if self.schedule.steps!=0:
            for agent in self.schedule.agents:
                #create entry for new time step and initialize it with updated value from previous timestep 
                self.attitude_dict[agent][self.schedule.steps]= []
                self.attitude_dict[agent][self.schedule.steps].append(self.attitude_dict[agent][self.schedule.steps - 1][-1])

                self.attitude_uncertainty_dict[agent][self.schedule.steps]= []
                self.attitude_uncertainty_dict[agent][self.schedule.steps].append(self.attitude_uncertainty_dict[agent][self.schedule.steps - 1][-1])

                self.pbc_dict[agent][self.schedule.steps]= []
                self.pbc_dict[agent][self.schedule.steps].append(self.pbc_dict[agent][self.schedule.steps - 1][-1])

                self.subnorms_dict[agent][self.schedule.steps]= []
                self.subnorms_dict[agent][self.schedule.steps].append(self.subnorms_dict[agent][self.schedule.steps - 1][-1])
        

        self.attitude_evolution()
        
        #pool = ProcessingPool(4)
        #results= pool.map(self.circles_of_influence, [interaction for interaction in interactions], chunksize=50) 
        self.subnorms_evolution() 

        ## after updating all three TPB attributes, calculate the intention with the latest values of tpb attributes
        for household in self.schedule.agents:
            household.attitude = self.attitude_dict[household][self.schedule.steps][-1]
            household.pbc = self.pbc_dict[household][self.schedule.steps][-1]
            household.subnorms = self.subnorms_dict[household][self.schedule.steps][-1]

            # calculating updated intention 
            household.intention = self.tpb_constant+ (self.pbc_weight * household.pbc) + (self.att_weight * household.attitude) + (self.sn_weight * household.subnorms)

            # check both thresholds:
            if self.schedule.steps <= 1: 
                if household.intention >= 0.80:
                    household.adoption_status = 1
            else:
                if (household.intention >= self.intention_threshold[self.schedule.steps]) and (household.pbc>=0.8):
                    household.adoption_status = 1 
            
            ## log everything into a dictionary which collects data:
            new_record = {'timestep':self.schedule.steps,'case_id':household.unique_id,'attitude':household.attitude,\
                            'subnorms':household.subnorms,'pbc':household.pbc,'intention':household.intention,'adoption_status':household.adoption_status,\
                            'tolerated_payback':household.ToleratedPayBackPeriod,'actualpayback':household.SimplePayBackPeriod,'geoid':household.geoid}
                            
            self.datacollector_df = self.datacollector_df.append(new_record, ignore_index=True)
        print('end of step', self.schedule.steps)
                    

################################################################################################################
#rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\Solar-Adoption-Agent-Based-Model\\'  #windows
rootpath = '/home/nfs/ameenakshisund/abm/Solar-Adoption-Model-ABM/'                                          #server 
#rootpath= '/Users/rtseinstein/Documents/GitHub/Solar-Adoption-Model-ABM/'                                       #mac 

#sample.step()
# can run upto 48 steps (4 years) 


def model_run(filename):
    seeded_df = pd.DataFrame()
    print(f'Executing for {filename[83:]}')   ## for server, it is filename[83:]. For mac it is: filename[90:]
    sample = AdoptionModel(filename)
    for i in range(8):
        sample.step()
    #rootpath= '/Users/rtseinstein/Documents/GitHub/Solar-Adoption-Model-ABM/'                                       #mac 
    rootpath = '/home/nfs/ameenakshisund/abm/Solar-Adoption-Model-ABM/'        
    outputfile = filename[83:]                              
    sample.datacollector_df.to_csv(rootpath+'experiment/integrated/scenario3b_1pp/'+str(outputfile))
    #seeded_df['seeded_agents']= sample.seeded_agents
    seeded_df.to_csv(rootpath+'experiment/integrated/scenario3b_1pp/seeds/'+str(outputfile))
    print(f'Finished exporting for {filename[83:]}')


#model_run(rootpath+'data/households_censustracts/tract_14203.csv')
filename= glob.glob(rootpath+'data/households_censustracts/*.csv')

##parallelizing runs
pool = ProcessingPool(10)
results = pool.map(model_run,filename)
