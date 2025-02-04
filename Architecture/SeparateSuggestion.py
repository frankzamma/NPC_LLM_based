from NPC.NPC import NPC
import random as rand
import numpy as np

class InjectionHelper:
    def __init__(self, npc : NPC):
        self.helper = npc

    def inject_suggestion(self, description_of_game_state, prob = 0):
        suggestion = -1

        if rand == 0 or rand.random() < prob:
            suggestion =  self.helper.get_advice(description_of_game_state)
        
        return suggestion



