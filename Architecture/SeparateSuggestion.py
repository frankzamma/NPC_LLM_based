from NPC.NPC import NPC
import random as rand
import numpy as np

class SeparateHelper:
    def __init__(self, npc : NPC):
        self.helper = npc

    def get_suggestion(self, description_of_game_state, prob = 0):
        action = -1

        if prob == 0 or rand.random() < prob: 
            suggestion =  self.helper.get_advice(description_of_game_state)
            print(f"- Description:\n{suggestion.description}\n")
            action = suggestion.action

        return action



