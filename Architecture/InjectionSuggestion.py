
from NPC.NPC import NPC
import random as rand
import numpy as np

class InjectionHelper:
    def __init__(self, npc : NPC):
        self.helper = npc

    def inject_suggestion(self, obs, description_of_game_state, prob = 0):
        action = -1

        if prob == 0 or rand.random() < prob:
            suggestion =  self.helper.get_advice(description_of_game_state)
            
            action = suggestion.action

        np.append(obs, action)

        return obs



