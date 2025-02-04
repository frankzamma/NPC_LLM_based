from abc import ABC, abstractmethod

class NPC(ABC):
    
    @abstractmethod
    def get_advice(self, env_state):
        pass