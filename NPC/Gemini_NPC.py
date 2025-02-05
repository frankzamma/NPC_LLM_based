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

        self.system_message = \
                    "You are an assitant of game player."\
                    "You receive the state of the game you suggest a possible strategies."\
                    "Your suggestion has two parts: description and action."\
                    "description is a brief descritpion of strategy, max 200 words."\
                    "action is a number beetween  0 and 8."\
                    

        
    def get_advice(self, env_state):

        client = genai.Client(api_key=self.key)

        response = client.models.generate_content(
            model='gemini-1.5-flash',
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
        return response

