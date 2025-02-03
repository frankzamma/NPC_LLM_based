# This file was extracted from the "Migliorare-un-NPC-in-un-gioco-a-turni-tramite-LLM-e-RL" repository.
# Original URL: https://github.com/Seldre99/Migliorare-un-NPC-in-un-gioco-a-turni-tramite-LLM-e-RL
# Source commit: 17aca5c
# Extraction date: January 25, 2025
# Original author: Andrea Selice(Seldre99)
# Original license: None



import random


class Spell:
    def __init__(self, name, cost, dmg, type):
        self.name = name
        self.cost = cost
        self.dmg = dmg
        self.type = type

    def generate_damage(self):
        low = self.dmg - 15
        high = self.dmg + 15
        return random.randrange(low, high)


