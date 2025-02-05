from tqdm import tqdm
from Algorithms.Agent_DQN import DQNAgent
from Algorithms.utils import salva_csv
from resources.game import Person
from resources.magic import Spell
from resources.inventory import Item
from resources.environment import BattleEnv
from NPC.Gemini_NPC import Gemini_NPC
from Architecture.SeparateSuggestion import SeparateHelper
import numpy as np
import torch

learning_rate = 0.01
n_episodes = 2
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  
final_epsilon = 0.1


# Spells and items setup
fire = Spell("Fire", 25, 600, "black")
thunder = Spell("Thunder", 30, 700, "black")
blizzard = Spell("Blizzard", 35, 800, "black")
meteor = Spell("Meteor", 40, 1000, "black")
cura = Spell("Cura", 32, 1500, "white")

potion = Item("Potion", "potion", "Heals 50 HP", 50)
hielixer = Item("MegaElixer", "elixer", "Fully restores party's HP/MP", 9999)
grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

player_spells = [fire, thunder, blizzard, meteor, cura]
player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                {"item": hielixer, "quantity": 1}]
player1 = Person("Valos", 3260, 132, 300, 34, player_spells, player_items)
enemy1 = Person("Magus", 4000, 701, 525, 25, [fire, cura], [])

players = [player1]
enemies = [enemy1]

env = BattleEnv(players, enemies)

obs = env.reset()

reward_per_episode = []
step_per_episode = []
epsilon_value = []

npc =  Gemini_NPC()

helper = SeparateHelper(npc)

state_dim = obs.shape[0]
action_dim = env.action_size

agent_base = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)
agent_base.model.load_state_dict(torch.load('models/model.pth'))

action_dim_agent = 2
agent = DQNAgent(state_dim, action_dim_agent, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)

# TRAINING
batch_size = 32
for episode in tqdm(range(n_episodes)):
    obs = env.reset()
    done = False

    total_reward = 0
    moves = 0
    enemy_choice = "No action"
    while not done:
        action_agent = agent_base.act(obs, False)
        action_npc =  helper.get_suggestion(env.describe_game_state(enemy_choice))

        actions = [action_agent, action_npc]

        selected_action =agent.act(obs, True)
        action = actions[selected_action]

        if selected_action == 0:
            print("Ho scelto RL")
        else:
            print("Ho scelto LLM")

        # print(f"episode:{episode}, steps:{moves} - azione selezionata")
        next_obs, reward, done, a_win, e_win, enemy_choice = env.step(action)

        # print(f"episode:{episode}, steps:{moves} - step eseguito")
        agent.remember(obs, action_npc, reward, next_obs, done)
        # print(f"episode:{episode}, steps:{moves} - remember eseguito")

        agent.replay(batch_size)
        # print(f"episode:{episode}, steps:{moves} - replay eseguito")

        obs = next_obs
        total_reward += reward
        moves +=1
       
        # print(env.describe_game_state(enemy_choice))

    reward_per_episode.append(total_reward)
    step_per_episode.append(moves)
    epsilon_value.append(agent.epsilon)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

#salva_csv(reward_per_episode, "Reward", "csv_reward_DQN.csv")
#salva_csv(step_per_episode, "Steps", "csv_steps_DQN.csv")
#salva_csv(epsilon_value, "Epsilon", "csv_epsilon_DQN.csv")

#agent.save("./models/model_LLM_2.pth" )