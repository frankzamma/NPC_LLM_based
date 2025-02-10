import os
import datetime
import re
from tqdm import tqdm
from Algorithms.Agent_DQN import DQNAgent
from Algorithms.utils import salva_csv
from resources.game import Person
from resources.magic import Spell
from resources.inventory import Item
from resources.environment import BattleEnv
from NPC.GPT4mini_NPC import GPT4mini_NPC
from Architecture.InjectionSuggestion import InjectionHelper
import numpy as np
import torch


n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  
final_epsilon = 0.1

# Probabilità di intervento del NPC
PROB = 0.5

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
agent_wins = []
enemy_wins = []
epsilon_value = []
success_rate = []
consigli_accettati = []
consigli_dati = []
total_agent_wins = 0

npc =  GPT4mini_NPC()

helper = InjectionHelper(npc)
obs = np.append(obs, 0)

state_dim = obs.shape[0]
action_dim = env.action_size

agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)
agent.model.load_state_dict(torch.load('./Results/Architecture1/GPT/Prob0.5/2025_02_09_01_12_08_369121/model_GPT_1.pth'))


dir = "./Results/Architecture1/GPT/Prob" + str(PROB) + "/testing"

if not(os.path.exists(dir)):
    os.mkdir(dir)

dir = dir + "/" + re.sub("\.|:|-| ", "_",str(datetime.datetime.now()))
if not(os.path.exists(dir)):
    os.mkdir(dir)

# TRAINING
batch_size = 32
for episode in tqdm(range(n_episodes)):
    obs = env.reset()
    done = False

    total_reward = 0
    moves = 0
    consigli_accettati_episode = 0
    consigli_dati_episode = 0

    obs = np.append(obs, -1)

    while not done:
        action = agent.act(obs, False)

        if obs[len(obs) -1] != -1:
            consigli_dati_episode +=1

        if action == obs[len(obs) - 1]:
            consigli_accettati_episode += 1
            print("- Decisione Agente: accettato consiglio\n\n")
        else:
            if obs[len(obs) -1] == -1:
                print("Consiglio NPC non dato\n\n")
            else:
                print("- Decisione Agente: ignorato consiglio\n\n")

        # print(f"episode:{episode}, steps:{moves} - azione selezionata")
        next_obs, reward, done, a_win, e_win, enemy_choice = env.step(action)

        describe_game_state = env.describe_game_state(enemy_choice)
        next_obs = helper.inject_suggestion(next_obs, describe_game_state, PROB)
        
        obs = next_obs
        total_reward += reward
        moves +=1

        if done:
            print(f"Episode: {episode}/{n_episodes}, Score: {total_reward}, Moves: {moves}, Epsilon: {agent.epsilon}")
            if a_win:
                agent_wins.append(1)
                enemy_wins.append(0)
                total_agent_wins += 1
            else:
                agent_wins.append(0)
                enemy_wins.append(1)

            success_rate.append(total_agent_wins / (episode + 1))
            print("Vittorie agente: ", agent_wins.count(1), " Vittorie nemico: ", enemy_wins.count(1))

       
        # print(env.describe_game_state(enemy_choice))

    reward_per_episode.append(total_reward)
    step_per_episode.append(moves)
    epsilon_value.append(agent.epsilon)
    consigli_accettati.append(consigli_accettati_episode)
    consigli_dati.append(consigli_dati_episode)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    if  episode % 50 == 0:
        dir_episode = dir + "/Episode_" + str(episode)

        if not(os.path.exists(dir_episode)):
            os.mkdir(dir_episode)
        
        salva_csv(reward_per_episode, "Reward", f"{dir_episode}/csv_reward_GPT.csv")
        salva_csv(step_per_episode, "Steps", f"{dir_episode}/csv_steps_GPT.csv")
        salva_csv(epsilon_value, "Epsilon", f"{dir_episode}/csv_epsilon_GPT.csv")
        salva_csv(agent_wins, "Agent_Win", f"{dir_episode}/csv_win_agent_GPT.csv")
        salva_csv(enemy_wins, "Enemy_Win", f"{dir_episode}/csv_win_enemy_GPT.csv")
        salva_csv(success_rate, "Success_Rate", f"{dir_episode}/csv_win_success_rate_GPT.csv")
        salva_csv(consigli_accettati, "Consigli_Accettati", f"{dir_episode}/csv_consigli_accettati_GPT.csv")
        salva_csv(consigli_dati, "Consigli_dati", f"{dir_episode}/csv_consigli_dati_GPT.csv")


salva_csv(reward_per_episode, "Reward", f"{dir}/csv_reward_GPT_FINAL.csv")
salva_csv(step_per_episode, "Steps", f"{dir}/csv_steps_GPT_FINAL.csv")
salva_csv(epsilon_value, "Epsilon", f"{dir}/csv_epsilon_GPT_FINAL.csv")
salva_csv(agent_wins, "Agent_Win", f"{dir}/csv_win_agent_GPT_FINAL.csv")
salva_csv(enemy_wins, "Enemy_Win", f"{dir}/csv_win_enemy_GPT_FINAL.csv")
salva_csv(success_rate, "Success_Rate", f"{dir}/csv_win_success_rate_GPT_FINAL.csv")
salva_csv(consigli_accettati, "Consigli_Accettati", f"{dir}/csv_consigli_accettati_GPT_FINAL.csv")
salva_csv(consigli_dati, "Consigli_dati", f"{dir}/csv_consigli_dati_GPT_FINAL.csv")

