## Parallelizing:
import pandas as pd 
import multiprocessing as mp 
import dill 
from pathos.multiprocessing import ProcessingPool 

def model_initialization():
    df = pd.read_csv(path+'4000_household_agents.csv')
    for agent in df:
        model.schedule.add(agent)

        #assign three circles of influence 
        agent.social_circle1 = social_circle1
        agent.social_circle2 = social_circle2
        agent.social_circle3 = social_circle3


def assign_interactions():
    for agent in schedule.agents:
        #geograhic neighbhours 
        neighbours = agent.get_neighbhours()
        interaction_list.append(agent,neighbhour)

        #interaction in circles of influence
        interaction_list.append(agent, social_circle1)
        interaction_list.append(agent, social_circle2)
        interaction_list.append(agent, social_circle3)

        return interaction_list

def attitude_change(agent1,agent2):
    #compare attitudes
    if agent1.attitude > agent2.attitude:
        # make some change to attitudes
        agent1.attitude -= 0.2
        agent2.attitude += 0.2 
    return agent1.attitude,agent2.attitude

def interactions(interaction):
        agent1 = interaction[0]
        agent2 = interaction[1]

        agent1.attitude,agent2.attitude = attitude_change(agent1,agent2)


def main():
    model_initialization()
    interaction_list=  assign_interactions()

    #pool = mp.Pool(10)
    pool = ProcessingPool(10)

    #interaction list can have over and above 89,000 interactions atleast 
    results = pool.map(interactions, [interaction for interaction in interaction_list])

main()

