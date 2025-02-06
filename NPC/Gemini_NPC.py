import os
import json
from .NPC import NPC
from google import genai
from google.genai import types
from dotenv import load_dotenv
from Architecture.Suggestion import Suggestion


class Gemini_NPC(NPC):
    def __init__(self):

        load_dotenv()

        self.key = os.getenv('GEMINI_KEY')
        #print("Key:" + str(self.key))

        # genai.configure(api_key=self.key)

        self.system_message = "You are a game asstitant for player."\
                    "The battle involves a player and an enemy. Both player and enemy are characterised by:"\
                    "HP (health point), MP (magic point), Attack points, Defence points, Available magic points and Available items. "\
                    "For spells, the use of MP is necessary, while items have limited availability. "\
                    "The player has 9 actions available: Attack (removes 300 HP from the enemy), which is of type Attack [0],"\
                    "Fire (removes 600 HP from the enemy at a cost of 25 MP of the player), which is of type Magic [1],"\
                    "Thunder (removes 700 HP from the enemy at a cost of 30 MP of the player), which is of type Magic [2], "\
                    "Blizzard (removes 700 HP from the enemy at a cost of 35 MP of the player), which is of type Magic [3], "\
                    "Meteor (removes 1000 HP from the enemy at a cost of 40 MP of the player), which is of type Magic [4], "\
                    "Cure (adds 1500 HP to the player at a cost of 32 MP of the player), which is of type Magic [5], "\
                    "Potion (adds 50 HP to the player), which is of type Item and the player has 3 [6], "\
                    "Grenade (removes 500 HP from the enemy), which is of type Item and the player has 2 [7], "\
                    "Elixir (restores all HP and MP points of the player), which is Item type and the player has 1 [8]."\
                    "The number beetween [] is the id of the action."\
                    "You receive the state of the game and you formulate a strategy for the player."\
                    "The suggestion has 2 part: description and action."\
                    "The description is a brief descritpion of proposed strategy, max 200 words."\
                    "Action is the action that you suggest."\
                    "The strategy must regarding only 1 action."
                    

        
    def get_advice(self, env_state):

        client = genai.Client(api_key=self.key)

        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=env_state,
            config=types.GenerateContentConfig(
                system_instruction= self.system_message,
                response_mime_type= 'application/json',
                response_schema= Suggestion,
            ),
        )

        #model = genai.GenerativeModel(model_name='gemini-1.5-flash',
         #                           system_instruction=self.system_message)

        #response = model.generate_content(self.system_message, 
        #config={
        #'response_mime_type': 'application/json',
        #'response_schema': list[Suggestion],
        #},)

        # print (response.text)

        #text =  response.text

        #text = text.replace('```json', '')
        #text = text.replace('```', '')

        #message_dict = json.loads(text)
        

        #print(message_dict["description"])
        return response.parsed

