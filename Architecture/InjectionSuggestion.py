
from NPC.NPC import NPC
import random as rand
import numpy as np

class InjectionHelper:
    def __init__(self, npc : NPC):
        self.helper = npc

    def inject_suggestion(self, obs, description_of_game_state, prob = 1):
        action = -1

        if prob == 1 or rand.random() < prob:
            suggestion =  self.helper.get_advice(description_of_game_state)
            print(f"- Description:\n{suggestion.description}\n")
            action = suggestion.action

        obs = np.append(obs, action)

        return obs



