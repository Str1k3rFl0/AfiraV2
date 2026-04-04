import random
import json

import os

class TestareAI:
    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), 'messageAI.json')
        with open(file_path, 'r') as f:
            self.jsonMessageAI = json.load(f)
            
    def random_message(self):
        message = random.choice(self.jsonMessageAI['message'])
        return message