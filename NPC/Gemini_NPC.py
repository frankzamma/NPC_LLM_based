import os
import json
from .NPC import NPC
import google.generativeai as genai
from dotenv import load_dotenv


class Gemini_NPC(NPC):
    def __init__(self):

        load_dotenv()

        self.key = os.getenv('GEMINI_KEY')
        #print("Key:" + str(self.key))

        genai.configure(api_key=self.key)

        self.system_message = \
                    "You are an assitant of game player."\
                    "You receive the state of the game you suggest a possible strategies."\
                    "Your suggestion must be formetted as JSON object with two fields: description and action."\
                    "description is a brief descritpion of strategy, max 200 words."\
                    "action is a number beetween  0 and 8."\
                    "Don't add ```json"

        
    def get_advice(self, env_state):

        model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                                    system_instruction=self.system_message)

        response = model.generate_content(self.system_message)
        print (response.text)

        text =  response.text

        text = text.replace('```json', '')
        text = text.replace('```', '')

        message_dict = json.loads(text)
        

        print(message_dict["description"])
        return message_dict["action"]

