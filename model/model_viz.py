import math
import random

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement

from model import AdoptionModel
from households import Household

"""
Run simulation with Visualization 
Print output at terminal
"""


# ------------------------------------------------------------
def network_portrayal(G):
    # The model ensures there is always 1 agent per node
    
    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color":"#FF0000" 
           
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": "#e8e8e8",
            "width": 2,
        }
        for (source, target) in G.edges
    ]

    return portrayal


network = NetworkModule(network_portrayal, 500, 500, library="d3")

# ---------------------------------------------------------------
"""
Launch the animation server 
Open a browser tab 
"""

canvas_width = 500
canvas_height = 500

server = ModularServer(
    AdoptionModel, [network], "Adoption Model" )

# The default port
server.port = 8080
server.launch()
