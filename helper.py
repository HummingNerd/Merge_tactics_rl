import cv2 as cv 
import numpy as np 

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

NAME_TO_ID = {v["name"]: k for k, v in CARD_TABLE.items()}
ID_TO_COST = {k: v["cost"] for k, v in CARD_TABLE.items()}
ID_TO_NAME = {k: v["name"] for k, v in CARD_TABLE.items()}


def update_elixir(observation_space, action):
    elixir = observation_space[-1]
    print(elixir)
    if action < 9:
        elixir += ID_TO_COST[observation_space[2*action]]*(2**(observation_space[2*action+1] - 1)) - 1
    else:
        level_profile = [0, 0, 0]
        for i in range(9):
            if observation_space[2*i] == observation_space[2*action]:
                level_profile[observation_space[2*i+1] - 1] = 1
        merge_elixir = 0
        if level_profile[0] == 1:
            merge_elixir = 2
            if level_profile[1] == 1:
                merge_elixir = 4
                if level_profile[2] == 1:
                    merge_elixir = 6
        print(level_profile)
        elixir += merge_elixir - observation_space[2*action + 1]
        
    return elixir

def single_TemplateMatch(needle, haystack, threshold=0.90):
    res = cv.matchTemplate(haystack, needle, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res)
    h, w = needle.shape[:2]

    if max_val >= threshold:
        return True, max_loc[0]//2+w/2, max_loc[1]//2+h/2
    return False, 0, 0

def hsv_masked_multi_TemplateMatch(needle, haystack, lower_bound = np.array([70, 80, 0]), upper_bound = np.array([102, 255, 255]), threshold=0.425, group_threshold = 1, eps = 0.2):
    needle_hsv = cv.cvtColor(needle, cv.COLOR_BGR2HSV)
    needle_mask = cv.inRange(needle_hsv, lower_bound, upper_bound)

    haystack_hsv = cv.cvtColor(haystack, cv.COLOR_BGR2HSV)
    haystack_mask = cv.inRange(haystack_hsv, lower_bound, upper_bound)

    res = cv.matchTemplate(haystack_mask, needle_mask, cv.TM_CCOEFF_NORMED)
        
    h, w = needle_mask.shape[:2]
    
    locations= np.where(res >= threshold)
    locations = list(zip(*locations[::-1]))

    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), w, h]
        rectangles.append(rect)
        rectangles.append(rect)

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.2)

    return rectangles

def slice_frame(img, region_main, region_slice):
    return img[2*(region_slice["top"] - region_main["top"]): 2*(region_slice["top"] - region_main["top"] + region_slice["height"]), 2*(region_slice["left"] - region_main["left"]): 2*(region_slice["left"] - region_main["left"] + region_slice["width"])]

def extract_shop(img, tempalte_images, template_images_name):
    number_cards_found = 0
    name_location_card = [None]*3 
    
    for i, image in enumerate(tempalte_images):
        detected, x, y = single_TemplateMatch(image, img, threshold=0.9)
        if detected:
            name_location_card[number_cards_found] = [template_images_name[i], x]
            number_cards_found += 1
            if number_cards_found == 3:
                break
    if number_cards_found < 3:
        return [None]*3
    cards = sorted(name_location_card,key=lambda x: x[1])
    cards = [card[0] for card in cards]
    return cards

def compute_action_mask(obs, COST_LOOKUP = ID_TO_COST):
    HAND_SLOTS = 9
    BUY_SLOTS = 3

    mask = []

    # -------- SELL actions (0–8) --------
    for i in range(HAND_SLOTS):
        card_id = obs[2*i]
        mask.append(card_id != -1)

    # -------- BUY actions (9–11) --------
    elixir = obs[-1]
    hand_full = -1 not in obs[:2*HAND_SLOTS:2]

    for i in range(BUY_SLOTS):
        card_id = obs[2*HAND_SLOTS + 2*i]

        if card_id == -1:
            mask.append(False)
            continue

        cost = COST_LOOKUP[card_id]

        if elixir < cost:
            mask.append(False)
            continue

        if hand_full and card_id not in obs[:2*HAND_SLOTS:2]:
            mask.append(False)
            continue

        mask.append(True)

    return np.array(mask, dtype=bool)

if __name__ == "__main__":
    pass