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
import os
import re
import datetime

n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  
final_epsilon = 0.1

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


npc =  Gemini_NPC()

helper = SeparateHelper(npc)

state_dim = obs.shape[0]
action_dim = env.action_size

agent_base = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)
agent_base.model.load_state_dict(torch.load('./Results/NormalAgent/Training/2025_02_06_11_13_16/model.pth'))

action_dim_agent = 2
agent = DQNAgent(state_dim, action_dim_agent, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)
agent.model.load_state_dict(torch.load('./Results/Architecture2/Gemini/Prob1/2025_02_09_11_53_39_229372/model_Gemini_2.pth'))


dir = "./Results/Architecture2/Gemini/Testing/Prob0.5"

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
    enemy_choice = "No action"
    while not done:
        action_agent = agent_base.act(obs, False)
        action_npc =  helper.get_suggestion(env.describe_game_state(enemy_choice))
        actions = [action_agent, action_npc]
        selected_action = 0
        
        if action_npc != -1:
            consigli_dati_episode += 1
            selected_action = agent.act(obs, True) # l'agente può scegliere il consiglio solo se l'ha ricevuto

        action = actions[selected_action]

        if selected_action == 0:
            if action_npc == -1:
                print("Consiglio NPC non dato")
            else:
                print("- Decisione Agente: ignorato consiglio\n\n")
        else:
            consigli_accettati_episode += 1
            print("- Decisione Agente: accettato consiglio\n\n")

        # print(f"episode:{episode}, steps:{moves} - azione selezionata")
        next_obs, reward, done, a_win, e_win, enemy_choice = env.step(action)

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
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    if  episode % 50 == 0:
        dir_episode = dir + "/Episode_" + str(episode)

        if not(os.path.exists(dir_episode)):
            os.mkdir(dir_episode)
        
        salva_csv(reward_per_episode, "Reward", f"{dir_episode}/csv_reward_Gemini.csv")
        salva_csv(step_per_episode, "Steps", f"{dir_episode}/csv_steps_Gemini.csv")
        salva_csv(epsilon_value, "Epsilon", f"{dir_episode}/csv_epsilon_Gemini.csv")
        salva_csv(agent_wins, "Agent_Win", f"{dir_episode}/csv_win_agent_Gemini.csv")
        salva_csv(enemy_wins, "Enemy_Win", f"{dir_episode}/csv_win_enemy_Gemini.csv")
        salva_csv(success_rate, "Success_Rate", f"{dir_episode}/csv_win_success_rate_Gemini.csv")
        salva_csv(consigli_accettati, "Consigli_Accettati", f"{dir_episode}/csv_consigli_accettati_Gemini.csv")
        salva_csv(consigli_dati, "Consigli_Dati", f"{dir_episode}/csv_consigli_dati_Gemini.csv")


salva_csv(reward_per_episode, "Reward", f"{dir}/csv_reward_Gemini_FINAL.csv")
salva_csv(step_per_episode, "Steps", f"{dir}/csv_steps_Gemini_FINAL.csv")
salva_csv(epsilon_value, "Epsilon", f"{dir}/csv_epsilon_Gemini_FINAL.csv")
salva_csv(agent_wins, "Agent_Win", f"{dir}/csv_win_agent_Gemini_FINAL.csv")
salva_csv(enemy_wins, "Enemy_Win", f"{dir}/csv_win_enemy_Gemini_FINAL.csv")
salva_csv(success_rate, "Success_Rate", f"{dir}/csv_win_success_rate_Gemini_FINAL.csv")
salva_csv(consigli_accettati, "Consigli_Accettati", f"{dir}/csv_consigli_accettati_Gemini_FINAL.csv")
salva_csv(consigli_dati, "Consigli_Dati", f"{dir}/csv_consigli_dati_Gemini_FINAL.csv")



