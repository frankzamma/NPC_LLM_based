import os
import json
from .NPC import NPC
from openai import OpenAI
from dotenv import load_dotenv



class GPT4mini_NPC(NPC):
    def __init__(self):
        # self.client = OpenAI()
        load_dotenv()
        self.key = os.getenv('OPENAI_KEY')
        self.client = OpenAI()
        # Setting dell'agent
        
        self.system_message = {"role": "developer", 
            "content": 
                    "You are an assitant of game player." 
                    "You receive the state of the game you suggest a possible strategies."
                    "Your suggestion must be formetted as JSON object with two fields: description and action"
                    "description is a brief descritpion of strategy, max 200 words"
                    "action is a number beetween  0 and 8."
                    "Don't add ```"}
        


    def get_advice(self, env_state):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                self.system_message, 
                {"role": "user", 
                 "content": env_state}]
        )
        message = completion.choices[0].message.content
        message_dict = json.loads(message)

        print(message_dict["description"])
        return message_dict["action"]

