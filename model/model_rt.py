from enum import unique
#from matplotlib.animation import TimedAnimation
from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import NetworkGrid
import random
from networkx.generators.small import house_graph
import pandas as pd
import networkx as nx
import multiprocessing as mp 

from households import Household


class AdoptionModel(Model):
    
    def __init__(self):

        self.schedule = BaseScheduler(self)
        self.space = None
        self.run_time = 8
        self.G = nx.Graph()

        ## Interaction Groups 
        self.block_dict = {}                        ## Geographic interaction
        self.incomegroup_dict= {}                   ## CIRCLE-3 (outermost) in Circle of Influence
        self.blockgroup_incomegroup_dict = {}       ## CIRCLE-2 in Circle of Influence
        #self.bothcircles=[]                         ## contains agents in both circles so that 6 of them can be picked to be part of the core-group 
        #self.coregroup = {}                         ## CIRCLE-1 contains the core social group of the household 
                                                    

        ## TPB attributes
        self.tpb_constant = 0.2799
        self.pbc_weight = 0.1409
        self.att_weight = 0.4717
        self.sn_weight = 0.1081  # TODO : substitute these values with ones from regression results : DONE 
        self.intention_threshold = 0.75


        self.datacollector_df = pd.DataFrame(columns = ['timestep','case_id','attitude','subnorms','pbc','adoption_status','geoid'])
        self.interactions_df = pd.DataFrame(columns=['timestep','first_agent','agent1_initial_attitude','agent1_final_attitude','second_agent','agent2_initial_attitude','agent2_final_attitude'])

        self.generate_model()

    def generate_model(self):
        """
        - Adds household agents from the file as Mesa Agents 
        - Adds these Mesa Household Agents as node in Graph G 
        - Block-wise households are stored in dictionary form

        """ 
        #rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\Solar-Adoption-Agent-Based-Model\\' 
        rootpath = '/home/nfs/ameenakshisund/abm/Solar-Adoption-Agent-based-Model/'

        #df = pd.read_csv(rootpath+'data\\households_subset\\subset_initialized_latlonvalues.csv')
        df = pd.read_csv(rootpath+'data/households_subset/subset_initialized_latlonvalues.csv')
        df = df.drop(columns='Unnamed: 0') 


        # create an empty dictionary for storing agents block-wise 
        for block in list(df['geoid'].unique()):
            self.block_dict[block] =[]

        # create an empty dictionary for storing agents in same income-group 
                                                            ### CIRCLE OF INFLUENCE : CIRCLE-2 
        for incomegroup in list(df['income'].unique()):
            self.incomegroup_dict[incomegroup]=[]

        for blockgroup in list(df['bgid'].unique()):
            for incomegroup in list(df['income'].unique()):
                self.blockgroup_incomegroup_dict[(blockgroup,incomegroup)] = [] 

        #####################################################
        #### INITIALIZING TOLERATED PAYBACK PERIOD  #########
        #####################################################

        pbc_dict = {}
        for _,row in df.iterrows():
            pbc_dict[row['CASE_ID']] = [row['pbc']]

        values = list(df['pbc'])
        min_value = min(values)
        max_value = max(values)

        for caseid in list(pbc_dict.keys()):
            fraction = ((pbc_dict[caseid][0] - min_value) / (max_value - min_value))*100

            pbc_dict[caseid].append(fraction)
            if 0<= fraction <1:
                pbc_dict[caseid].append(0)
            if 1<= fraction <5:
                pbc_dict[caseid].append(3)
            if 5<= fraction <10: 
                pbc_dict[caseid].append(5)
            if 10<= fraction <25: 
                pbc_dict[caseid].append(6)
            if 25<= fraction <35: 
                pbc_dict[caseid].append(7.5)
            if 35<= fraction <40: 
                pbc_dict[caseid].append(8)
            if 40<= fraction <50: 
                pbc_dict[caseid].append(10)
            if 50<= fraction <78: 
                pbc_dict[caseid].append(15)
            if 78<= fraction <95: 
                pbc_dict[caseid].append(20)
            if 95<= fraction <100: 
                pbc_dict[caseid].append(25)
            if fraction== 100: 
                pbc_dict[caseid].append(30)

        tolerated_paybackperiods= []
        for i in list(pbc_dict.values()):
            tolerated_paybackperiods.append(i[2])
        
        # add this column to df
        df['toleratedpayback']= tolerated_paybackperiods
        #print(tolerated_paybackperiods)

        for _,row in df.iterrows():
            agent = Household(unique_id = str(row['CASE_ID']),
                             model = self, 
                             income = row['income'],
                             age= row['age'],
                             size= row['household_'],
                             ami_category = row['ami_categ'],
                             elec_consumption= row['elec_consumption'],
                             attitude = row['attitude'],
                             attitude_uncertainty = 1-abs(row['attitude']),
                             pbc = row['pbc'],
                             subnorms = row['subnorms'],
                             geoid = row['geoid'],
                             tract = row['tract'],
                             bgid = row['bgid'],
                             ToleratedPayBackPeriod= row['toleratedpayback'],
                             circle1=[],
                             circle2=[],
                             circle3=[],
                             adoption_status = 0)

            if agent:
                #print('agent',agent.unique_id,'initial_pbc:',agent.pbc,'with tolerated payback of',agent.ToleratedPayBackPeriod)

                self.schedule.add(agent)
                self.G.add_node(agent, label=agent.unique_id)   # main graph holds all agent nodes

                self.block_dict[agent.geoid].append(agent)
                self.incomegroup_dict[agent.income].append(agent)
                self.blockgroup_incomegroup_dict[(agent.bgid,agent.income)].append(agent)
                
        # verifying if this was implemented correctly    
        #print('block dictionary: \n', self.block_dict)
        #print('total number of unique block-ids:\n ', len(list(df['geoid'].unique())))

        #######################################
        ######## CIRCLES OF INFLUENCE #########
        #######################################
    
        # after all the agents have been initialized, now give them coregroups
        for agent in self.schedule.agents:
            print('\n agent name:', agent.unique_id)
            #print(self.blockgroup_incomegroup_dict[(agent.bgid,agent.income)])
            #print(self.incomegroup_dict[agent.income])
            circle2 = self.blockgroup_incomegroup_dict[(agent.bgid,agent.income)]     # TODO: Remove the agent from their own circles of influence. 
            circle3 = self.incomegroup_dict[agent.income]                         # returns a list of the members of the household's incomegroup bracket. 
        
            #print(circle2)
            #print(circle3)
            if circle2 is not None:
                circle2_choices = list(random.choices(circle2,k=min(len(circle2),3)))
                circle2 = [x for x in circle2 if x not in circle2_choices]
                agent.circle2 = circle2
            else:
                circle2_choices=[] 

            if circle3 is not None:
                circle3_choices = list(random.choices(circle3,k=min(len(circle3),3)))
                circle3 = [x for x in circle3 if x not in circle3_choices]
                agent.circle3 = circle3
            else:
                circle3_choices = [] 

            agent.circle1= circle2_choices+circle3_choices  ## circle1 
            
            print('agent circle-1: ', len(agent.circle1))
            #print('members of circle1:',[x.unique_id for x in list(agent.circle1)])
            print('second circle:', len(agent.circle2))
            #print('members of circle2:',[x.unique_id for x in list(agent.circle2)])
            print('third circle', len(agent.circle3)) 
            #print('members of circle3:',[x.unique_id for x in list(agent.circle3)])
             
             

    def attitude_evolution(self):
        """
        Using the relative aggreement algorithm, model interactions between agents at:
        - block level (physical interactions bound by geography)
        - socioeconomic level (interactions within different circles of the agent)
        """

        ## for every block in block_dict:
        ### a) get list of agents in that block. 
        ### b) add them to a temporary graph temp_G and add random edges between them
        ### c) compose them to a main graph called maingraph  (needed?)
        ### d) add the new edges to a global list containing all interactions in that time-step
        ###    this will be refreshed at every time-step beginning 

        ### e) for every pair in the list of interactions for that time-step, run the RA_algo

        ### Two functions are required:
        #   (i) add_and_remove_edges 
        #   (ii) ra_algorithm
        
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
          
        ## Source:  http://jasss.soc.surrey.ac.uk/15/4/4.html
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
            x_i= a.attitude
            #print('initial x_i', x_i)
            u_i= a.attitude_uncertainty   ## DONE: this also has to be initialized at the very beginning for this to change. 

            x_j= b.attitude
            #print('initial x_j', x_j)
            u_j= b.attitude_uncertainty 



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

                if b.adoption_status == 1:          #so that opinions change only for non-adopters. 
                    b.attitude = b.attitude         #this means 
                    b.attitude_uncertainty = b.attitude_uncertainty
                else:
                    b.attitude = x_j 
                    b.attitude_uncertainty = u_j   #updating uncertainty as well. 
                
            return a.attitude,b.attitude

        #maingraph= nx.Graph()
        interactions = []

        ######################
        ## GEOGRAPHIC ########
        ######################

        ## creating random interactions between agents within a census block [geographically]

                    ## This implies that not all agents interact visually with members of the neigbhourdhood every month. 
                    ## Some agents may not interact with any other agent during a month. 
        for block in list(self.block_dict.keys()):
            tempG = nx.Graph() ## temp graph to store the households in a block
            householdagents_in_block= self.block_dict[block]
            
            for household in householdagents_in_block:
                tempG.add_node(household)
            # graph created
            removed_edges,added_edges = add_and_remove_edges(tempG,p_new_connection=0.6,p_remove_connection=0.6)
                                                                                ## TODO: be sure to change these values for Sensitivity Analysis?
            for newedge in added_edges:
                interactions.append(newedge)
        
        
        ## creating interactions between blocks. People interact outside the block if they belong to the same income group
        
        ######################
        #### SOCIOECONOMIC ###
        ######################

        ## Here every agent will be activated to randomly interact with members of their inner group, workplace group and finally circle-3
        ## circle-3 can be used to simulate online social network activities, although the belief that people attach from information originating from such networks 
                ## will be modelled to be low. 

        ## because every time-step, every agent is activated for this interaction, we will loop through every agent in the schedule. 
            ## if they are adopters, they will influence and not be influenced. if the agent they are interacting with also is an adopter, there will be no change in opinions. 
            ## if they are not yet adopters, they will BE influenced by their circles. they will be the "SECOND Agent (B)" in the relative agreement theory. 

        ## for every agent, get the members of their three circles and draw path graphs between them. add these to the main interactions list.

        for agent in self.schedule.agents:
            ## FIRST CIRCLE ## 
            # core group for every household
            members_circle1 = agent.circle1   # DONE: does not contain agent. #removing agent so that circle members contains only neighs with whom the agent will interact.
            random_members_circle1 = random.choices(members_circle1, k=int(len(members_circle1)/2))   ## interact with 3 of the coregroup members each round
            # create a graph                                                                          ## TODO: Key Parameter
            tempG = nx.Graph()
            tempG.add_node(agent)
            for member in members_circle1:
                if agent!=member:
                    tempG.add_node(member)

            tempG.add_edges_from([(agent,node) for node in random_members_circle1])
            for circle1_interaction  in list(tempG.edges()):
                interactions.append(circle1_interaction)


            ## SECOND CIRCLE ## 
            members_circle2 = agent.circle2 
            random_members_circle2 = random.choices(members_circle2, k=int(len(members_circle2)*0.20))   ## interact with 20% of circle3 members
                                                                                                        ## TODO: Key Parameter
            # create a graph 
            tempG = nx.Graph()
            tempG.add_node(agent)
            for member in members_circle2:
                if agent!=member:
                    tempG.add_node(member)

            tempG.add_edges_from([(agent,node) for node in random_members_circle2])
            for circle2_interaction  in list(tempG.edges()):
                interactions.append(circle2_interaction)
            

           ## THIRD CIRCLE ## 
            members_circle3 = agent.circle3
            random_members_circle3 = random.choices(members_circle3, k=int(len(members_circle3)*0.01))   ## interact with 1% of circle3 members
                                                                                                        ## TODO: Key Parameter
            # create a graph 
            tempG = nx.Graph()
            tempG.add_node(agent)
            for member in members_circle3:
                if agent!=member:
                    tempG.add_node(member)

            tempG.add_edges_from([(agent,node) for node in random_members_circle3])
            for circle3_interaction  in list(tempG.edges()):
                interactions.append(circle3_interaction)           


        
        def circles_of_influence(interaction):
            ## making it directional on adopters # TODO: just marking for easy access. 
            if (interaction[0].adoption_status ==1) and (interaction[1].adoption_status==1): #if both agents are adopters, no need RA-implementation
                print('Time step:', self.schedule.steps,',both', interaction[0].unique_id,'and',interaction[1].unique_id,'are adopters, no change in opinions!')
            if (interaction[0].adoption_status==1) and (interaction[1].adoption_status==0): 
                first_agent = interaction[0]
                second_agent = interaction[1]
                first_agent_initial_attitude = first_agent.attitude
                second_agent_initial_attitude= second_agent.attitude 
                first_agent.attitude, second_agent.attitude = ra_implementation(first_agent,second_agent)

                ## datacollection 
                new_record = {'timestep':self.schedule.steps, 'first_agent':first_agent.unique_id, 'agent1_initial_attitude': first_agent_initial_attitude, 'agent1_final_attitude':first_agent.attitude, 'second_agent':second_agent.unique_id,'agent2_initial_attitude':second_agent_initial_attitude,'agent2_final_attitude':second_agent.attitude}
                self.interactions_df = self.interactions_df.append(new_record,ignore_index=True)
            else:
                first_agent = interaction[1]
                second_agent= interaction[0]

                first_agent_initial_attitude = first_agent.attitude
                second_agent_initial_attitude= second_agent.attitude 

                first_agent.attitude, second_agent.attitude = ra_implementation(first_agent,second_agent)
                ## datacollection 
                new_record = {'timestep':self.schedule.steps, 'first_agent':first_agent.unique_id, 'agent1_initial_attitude': first_agent_initial_attitude, 'agent1_final_attitude':first_agent.attitude, 'second_agent':second_agent.unique_id,'agent2_initial_attitude':second_agent_initial_attitude,'agent2_final_attitude':second_agent.attitude}
                self.interactions_df = self.interactions_df.append(new_record,ignore_index=True)     

        
        pool = mp.Pool(2)
        results = pool.map(circles_of_influence, [interaction for interaction in interactions]) 
        #print('first agent', first_agent.unique_id, 'second agent', second_agent.unique_id)
        ##verifying implementation
        #print('new attitudes of last pair:', first_agent.attitude, 'and', second_agent.attitude)
         
        ## add interactions to a datacollector dataframe 
        ## should be <time-step> <first-agent> <second-agent> <initial-attitude-agent-1> <final-attitude-agent-1> <initial-attitude-agent2> <final-attitude-agent2>
      
    def subnorms_evolution(self):
        """
        Modelling the influence of adopters in the block, on adoption decisions of other block residents

        """ 
        for household in self.schedule.agents:
            # get the number of adopters in the household's block
            adopters_in_block = []
            for hh in self.block_dict[household.geoid]:
                if hh.adoption_status==1:
                    adopters_in_block.append(hh)
            if household.adoption_status==0:
                if len(adopters_in_block) > (0.33 * len(self.block_dict[household.geoid])): 
                    household.subnorms = min(household.subnorms+0.5,1)       
                                            ## if more than one-third of the neighbors have a solar panel on their roof, 
                                            ## the household's subnorms becomes increases by 0.5 
                                            ## any scientific explanation for the threshold? 
                if len(adopters_in_block) > (0.5 * len(self.block_dict[household.geoid])): 
                    household.subnorms = 1                 
                                            ## if more than 1/2 of the neighbors have a solar panel on their roof, 
                                            ## the household's subnorms becomes increases to 1 
                                            ## any scientific explanation for the threshold? 
        

    def step(self):
        """
        Advance the model by a step
        """
        self.attitude_evolution()
        
        self.subnorms_evolution() 

        #PBC is performed at the agent level 

        self.schedule.step() # advancing the state of all agents by one.

        ## after updating all three TPB attributes, calculate the intention:
        for household in self.schedule.agents:
            # calculating updated intention 
            household.intention = self.tpb_constant+ (self.pbc_weight * household.pbc) + (self.att_weight * household.attitude) + (self.sn_weight *household.subnorms)

            # check both thresholds:
            if (household.intention >= self.intention_threshold) and (household.pbc>=0.8):
                household.adoption_status = 1 
            
            ## log everything into a dictionary which collects data:
            new_record = {'timestep':self.schedule.steps,'year':household.timestep_to_year(),'case_id':household.unique_id,'attitude':household.attitude,\
                            'subnorms':household.subnorms,'pbc':household.pbc,'intention':household.intention,'adoption_status':household.adoption_status,\
                            'tolerated_payback':household.ToleratedPayBackPeriod,'actualpayback':household.SimplePayBackPeriod,'geoid':household.geoid}
                            
            self.datacollector_df = self.datacollector_df.append(new_record, ignore_index=True)



################################################################################################################
#rootpath = 'c:\\Users\\Gamelab\\Desktop\\RT\\Others\\Thesis\\Thesis_coding\\ABM\\Solar-Adoption-Agent-Based-Model\\' 
rootpath = '/home/nfs/ameenakshisund/abm/Solar-Adoption-Agent-based-Model/'

sample = AdoptionModel()
#sample.step()
# can run upto 48 steps (4 years) 
for i in range(36):
    sample.step()

#sample.datacollector_df.to_csv(rootpath+'experiment\\output_mainsubset_36runs_corrected.csv')
#sample.interactions_df.to_csv(rootpath+'experiment\\interactions_mainsubset_36runs_corrected.csv')

sample.datacollector_df.to_csv(rootpath+'experiment/server/output_mainsubset_12runs_corrected.csv')
sample.interactions_df.to_csv(rootpath+'experiment/server/interactions_mainsubset_12runs_corrected.csv')
