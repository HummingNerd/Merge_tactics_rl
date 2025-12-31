import os 
import cv2 as cv
import numpy as np
from mss import mss
import pyautogui
import time
import random
from helper import *
from sb3_contrib import MaskablePPO
from model_loader import load_resnet, predict_crop

# --------------------------------------------------
# 🧠 STRATEGY MODULE
# --------------------------------------------------

def get_hand_and_shop(obs):
    """Parses the flat observation list into structured data."""
    # Hand: indices 0-17 (pairs of id, level)
    hand = [] 
    for i in range(0, 18, 2):
        cid, lvl = obs[i], obs[i+1]
        hand.append((cid, lvl))
        
    # Shop: indices 18-23 (pairs of id, cost)
    shop = []
    for i in range(18, 24, 2):
        cid, cost = obs[i], obs[i+1]
        if cid == -1:
            shop.append(None)
        else:
            shop.append(cid)
            
    elixir = obs[-1]
    return hand, shop, elixir

def get_total_wealth(hand, elixir):
    """Calculates liquid elixir + inventory value."""
    inventory_val = 0
    for cid, lvl in hand:
        if cid != -1 and lvl > 0:
            # Value estimation: Cost * 2^(lvl-1)
            # You can adjust this to match your game's exact sell prices
            base_cost = ID_TO_COST.get(cid, 0)
            inventory_val += base_cost * (2**(lvl-1)) - 1
    return elixir + inventory_val

def heuristic_policy_1(obs):
    hand, shop, elixir = get_hand_and_shop(obs)

    # 1. Sell any Level 2 immediately
    for i, (cid, lvl) in enumerate(hand):
        if cid != -1 and lvl == 2:
            return i  # Sell

    # 2. Buy merge if possible
    for i, card in enumerate(shop):
        if card is None:
            continue
        cost = ID_TO_COST[card]
        if elixir >= cost and (card, 1) in hand:
            return 9 + i  # Buy merge

    # 3. Buy cheapest available card
    cheapest_idx = None
    cheapest_cost = float("inf")

    for i, card in enumerate(shop):
        if card is None:
            continue
        cost = ID_TO_COST[card]
        if cost < cheapest_cost:
            cheapest_cost = cost
            cheapest_idx = i

    if cheapest_idx is not None and elixir >= cheapest_cost:
        return 9 + cheapest_idx

    return None

def heuristic_policy_2(obs):
    hand, shop, elixir = get_hand_and_shop(obs)

    # 0. Sell any Level 2 immediately (highest priority)
    for i, (cid, lvl) in enumerate(hand):
        if cid != -1 and lvl == 2:
            return i  # Sell

    # 1. Buy merge if possible
    for i, card in enumerate(shop):
        if card is None:
            continue
        cost = ID_TO_COST[card]
        if elixir >= cost and (card, 1) in hand:
            return 9 + i

    # 2. Buy cheapest card
    cheapest_idx = None
    cheapest_cost = float("inf")

    for i, card in enumerate(shop):
        if card is None:
            continue
        cost = ID_TO_COST[card]
        if cost < cheapest_cost:
            cheapest_cost = cost
            cheapest_idx = i

    if cheapest_idx is not None and elixir >= cheapest_cost:
        return 9 + cheapest_idx

    # 3. Sell most expensive Level 1 card
    worst_idx = None
    highest_cost = -1

    for i, (cid, lvl) in enumerate(hand):
        if cid != -1 and lvl == 1:
            cost = ID_TO_COST.get(cid, 0)
            if cost > highest_cost:
                highest_cost = cost
                worst_idx = i

    if worst_idx is not None:
        return worst_idx  # Sell

    return None

def sell(start_pos):
    pyautogui.moveTo(start_pos)
    pyautogui.mouseDown()
    pyautogui.dragTo(1050, 665, duration=0.15, button="left")
    # time.sleep(0.1)
    pyautogui.mouseUp()
    # time.sleep(0.1)
    pyautogui.moveTo(1050, 250)

def buy(idx):
    # Calculate the target X position
    target_x = 1060 + 60 * (idx - 2)
    target_y = 650

    # First Click: Select the item
    pyautogui.click(target_x, target_y)
    time.sleep(0.05)
    pyautogui.click(target_x, target_y)

    # Move out of the way
    # Wait a moment so the move-away doesn't "smear" the click
    pyautogui.moveTo(1050, 150)

# -----------------------
# Config
# -----------------------
scale = 0.5
screen_area = {"left": 880, "top": 58, "width": 370, "height": 650}
battle_area = {"left": 885, "top": 145, "width": 50, "height": 20}
troop_area = {"left": 940, "top": 375, "width": 250, "height": 200}
shop_area = {"left" : 970, "top" : 630, "width" : 175, "height" : 60}

LEVEL_MODEL_PATH = "models/level_net.pth"
TROOP_MODEL_PATH = "models/troop_net.pth"

LEVEL_CLASSES = [1,3,2] # ['bronze', 'gold', 'silver']
TROOP_CLASSES = ['archer_queen', 'bandit', 'barbarians', 'dart_goblin', 'electro_giant', 'executioner', 'goblin_machine', 'goblins', 'golden_knight', 'mega_knight', 'mini_pekka', 'monk', 'musketeer', 'pekka', 'prince', 'princess', 'royal_ghost', 'royal_giant', 'skeleton_dragons', 'skeleton_king', 'spear_goblins', 'valkyrie', 'witch', 'wizard']
 
 
level_model = load_resnet(
     checkpoint_path=LEVEL_MODEL_PATH,
     conf_thresh=0.9 
 )
troop_model = load_resnet(
     checkpoint_path=TROOP_MODEL_PATH,  
     conf_thresh=0.9      
 )

# -----------------------
# Load & preprocess NEEDLE (ONCE)
# -----------------------
NEEDLE = cv.resize(cv.imread("assets/needle.png"), (0, 0), fx=0.5, fy=0.5)
NEEDLE_HEIGHT, NEEDLE_WIDTH = NEEDLE.shape[:2]

DEPLOY_PHASE = cv.imread("assets/deploy_phase.png")

TRANSITION= cv.resize(cv.imread("assets/transition.png"), (0, 0), fx=0.5, fy=0.5)

IMG_DIR = "assets/shop_images"

shop_images = []      # list of image arrays
shop_filenames = []   # matching filenames

for fname in sorted(os.listdir(IMG_DIR)):
    path = os.path.join(IMG_DIR, fname)

    img = cv.imread(path)
    if img is None:
        continue  # skip non-images or failed loads

    shop_images.append(img)
    shop_filenames.append(fname[:-4])

# Canonical card table
# ID is the single source of truth across:
# - classifiers
# - observation encoding
# - action space
# - reward logic

CARD_TABLE = {
    0:  {"name": "goblins",            "cost": 2},
    1:  {"name": "spear_goblins",      "cost": 2},
    2:  {"name": "mini_pekka",        "cost": 2},
    3:  {"name": "barbarians",         "cost": 2},
    4:  {"name": "skeleton_dragons",   "cost": 2},
    5:  {"name": "wizard",            "cost": 2},
    6:  {"name": "royal_giant",       "cost": 2},

    7:  {"name": "musketeer",         "cost": 3},
    8:  {"name": "valkyrie",          "cost": 3},
    9:  {"name": "pekka",             "cost": 3},
    10: {"name": "prince",            "cost": 3},
    11: {"name": "dart_goblin",       "cost": 3},
    12: {"name": "electro_giant",     "cost": 3},
    13: {"name": "executioner",       "cost": 3},

    14: {"name": "witch",             "cost": 4},
    15: {"name": "princess",          "cost": 4},
    16: {"name": "mega_knight",       "cost": 4},
    17: {"name": "royal_ghost",       "cost": 4},
    18: {"name": "bandit",            "cost": 4},
    19: {"name": "goblin_machine",    "cost": 4},

    20: {"name": "skeleton_king",     "cost": 5},
    21: {"name": "golden_knight",     "cost": 5},
    22: {"name": "archer_queen",      "cost": 5},
    23: {"name": "monk",              "cost": 5}
}

elixir = 4
NAME_TO_ID = {v["name"]: k for k, v in CARD_TABLE.items()}
ID_TO_COST = {k: v["cost"] for k, v in CARD_TABLE.items()}
ID_TO_NAME = {k: v["name"] for k, v in CARD_TABLE.items()}

# just debuging shit 
first_occurance_of_shop = True

prev_shop = [None]*3

PPO_MODEL_PATH = "models/best_model.zip"

ppo_model = MaskablePPO.load(PPO_MODEL_PATH)

round = 1

sct = mss()

# time.sleep(5)

while pyautogui.position()[1] > 25:
    # -----------------------
    # Capture frame
    # -----------------------
    screen_frame = np.array(sct.grab(screen_area))

    screen_frame_copy = screen_frame.copy()

    screen_frame = cv.cvtColor(screen_frame, cv.COLOR_BGRA2BGR)

    


    battle_frame = slice_frame(screen_frame, screen_area, battle_area)
    troop_frame = cv.resize(slice_frame(screen_frame, screen_area, troop_area), (0, 0), fx=0.5, fy=0.5)
    shop_frame = slice_frame(screen_frame, screen_area, shop_area)

    in_transition = single_TemplateMatch(TRANSITION, troop_frame, threshold=0.95)
    if in_transition[0]:
        elixir += 4
        round += 1
        time.sleep(1)
        shop = [None]*3
        continue


    in_deploy_phase, _, _ = single_TemplateMatch(DEPLOY_PHASE, battle_frame, threshold=0.99)
    if in_deploy_phase:
        # print("detected deploy phase")
        shop = extract_shop(shop_frame, shop_images, shop_filenames)
        if (shop != prev_shop or True) and shop != [None]*3:
            # print("detected shop")
            hp_bars = hsv_masked_multi_TemplateMatch(NEEDLE, troop_frame)
            if first_occurance_of_shop:
                first_occurance_of_shop = False
            else:
                if (True or len(hp_bars) != 0):
                            # print("detected troops")
                            observation_space = []
                            levels = []
                            troops = []
                            troop_positions = []
                            for i, (x,y,w,h) in enumerate(hp_bars):
                                position = (2*(troop_area["left"] - screen_area["left"] + x + int(w/2)), 2*(troop_area["top"] - screen_area["top"] + y + int(h/2)))
                                
                                cv.drawMarker(screen_frame, (position[0], position[1]), (0, 0, 255), cv.MARKER_CROSS)
                                level_off_set = (-23, -6)
                                level_size = (10, 10)
                                
                                # cv.rectangle(frame, top_left, bottom_right, line_color, 1, line_type)
                                level_img = screen_frame_copy[position[1] + 2*level_off_set[1]: position[1] + 2*level_off_set[1] + 2*level_size[1], position[0] + 2*level_off_set[0]: position[0] + 2*level_off_set[0] + 2*level_size[0]]
                                # level_img = cv.cvtColor(level_img, cv.COLOR_BGRA2RGB)
                                level_label, level_conf = predict_crop(level_model, level_img)
                                if level_label == "bronze":
                                    level_label = 1
                                elif level_label == "silver":
                                    level_label = 2
                                elif level_label == "gold":
                                    level_label = 3
                                else:
                                    continue
                                cv.imwrite(f"debug/{level_label}_{level_conf}.png", level_img)
                                levels.append(level_label)
                                cv.rectangle(screen_frame, (position[0] + 2*level_off_set[0], position[1] + 2*level_off_set[1]), (position[0] + 2*(level_off_set[0] + level_size[0]), position[1] + 2*(level_off_set[1] + level_size[1])), (0, 0, 255), 2, cv.LINE_4)
                                troop_off_set = (int(-w/2), +10)
                                troop_size = (w, 40)
                                troop_img = screen_frame_copy[position[1] + 2*troop_off_set[1]: position[1] + 2*troop_off_set[1] + 2*troop_size[1], position[0] + 2*troop_off_set[0]: position[0] + 2*troop_off_set[0] + 2*troop_size[0]]
                                # troop_img = cv.cvtColor(troop_img, cv.COLOR_BGRA2RGB)
                                troop_label, troop_conf = predict_crop(troop_model, troop_img)
                                if troop_label == "UNCERTAIN":
                                    continue
                                cv.imwrite(f"debug/{troop_label}_{troop_conf}.png", troop_img)
                                troop_positions.append((troop_area["left"] + x + int(w/2),
                                            troop_area["top"] + y + 30))
                                
                                troops.append(troop_label)
                                # troop_positions.append((screen_area["left"] + position[0], position[1] + 20))
                                if troop_label == "EMPTY":
                                    observation_space.append(-1)
                                    observation_space.append(-1)
                                else:
                                    observation_space.append(NAME_TO_ID[troop_label])
                                    observation_space.append(level_label)
                            # for i in range(len(levels)):
                                # print(f"{NAME_TO_ID[troops[i]]}, {troops[i]} : {levels[i]}")
                            # print(shop)
                            # if len(troop_positions):
                            #     print(troop_positions[0][0], troop_positions[0][1])
                            print("="*30)
                            while len(observation_space) < 18:
                                observation_space.append(-1)
                            observation_space = observation_space[:18]
                            while len(observation_space) > 18:
                                observation_space.pop()
                            for card in shop:
                                observation_space.append(NAME_TO_ID[card])
                                observation_space.append(ID_TO_COST[NAME_TO_ID[card]])
                            observation_space.append(elixir)
                            numpy_observation_space = np.array(observation_space, dtype=np.int32)
                            prev_shop = shop.copy()
                            action_mask = compute_action_mask(numpy_observation_space)

                            ### Action Space
                            hand_temp, shop_temp, elixir_temp = get_hand_and_shop(observation_space) 
                            wealth_temp = get_total_wealth(hand_temp, elixir_temp)
                            # print(wealth_temp)
                            if round < 3:
                                # print("Heur1")
                                action = heuristic_policy_1(observation_space)
                            else:
                                if wealth_temp < 15:
                                    # print("Heur2")
                                    action = heuristic_policy_2(observation_space)
                                else:
                                    # print("PPO")
                                    action, _ = ppo_model.predict(
                                        numpy_observation_space,
                                        action_masks=action_mask,
                                        deterministic=True
                                    )
                            if action == None:
                                prev_shop = [None, None, None]
                                continue
                            else:
                                elixir = update_elixir(observation_space, action)
                                if action < 9:
                                    sell(troop_positions[action])
                                    prev_shop = [None, None, None]
                                else:
                                    # print("BUY")
                                    buy(action%9 + 1)       
                            # first_occurance_of_shop = True
                            time.sleep(0.2) # Wait for update of screen
                            print(observation_space)
                            print(action)
    cv.imshow("sliced", troop_frame)
    if cv.waitKey(1) == ord("q"):
        break


cv.destroyAllWindows()