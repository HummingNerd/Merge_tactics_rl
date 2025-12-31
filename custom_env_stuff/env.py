import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import random 
import math

"New Reward "
"Configure Everything Once "

CARD_TABLE = {
            0: {"name": "Goblin", "cost": 2},
            1: {"name": "Spear_Goblin", "cost": 2},
            2: {"name": "Mini_Pekka", "cost": 2},
            3: {"name": "Barbarian", "cost": 2},
            4: {"name": "Skeleton_Dragon", "cost": 2},
            5: {"name": "Wizard", "cost": 2},
            6: {"name": "Royal_Giant", "cost": 2},
            7: {"name": "Musketeer", "cost": 3},
            8: {"name": "Valkyrie", "cost": 3},
            9: {"name": "Pekka", "cost": 3},
            10: {"name": "Prince", "cost": 3},
            11: {"name": "Dart_Goblin", "cost": 3},
            12: {"name": "Electro_Giant", "cost": 3},
            13: {"name": "Executioner", "cost": 3},
            14: {"name": "Witch", "cost": 4},
            15: {"name": "Princess", "cost": 4},
            16: {"name": "Mega_Knight", "cost": 4},
            17: {"name": "Royal_Ghost", "cost": 4},
            18: {"name": "Bandit", "cost": 4},
            19: {"name": "Goblin_Machine", "cost": 4},
            20: {"name": "Skeleton_King", "cost": 5},
            21: {"name": "Golden_Knight", "cost": 5},
            22: {"name": "Archer_Queen", "cost": 5},
            23: {"name": "Monk", "cost": 5}     
        }



def hand_strength(hand, card2cost):
    reward = 0
    for i in range(len(hand)):
        if hand[i][0] == -1:
            continue
        reward += math.sqrt(card2cost[hand[i][0]])*(2**(hand[i][1] - 1))

    return reward


def update_shop(deck, TOTAL_CARDS):
    if sum(1 for num in deck if num != 0) < 3:
        return [None, None, None]
    card1 = random.randint(0, TOTAL_CARDS-1)
    while deck[card1] <= 0:
        card1 = random.randint(0, TOTAL_CARDS-1)
    card2 = random.randint(0, TOTAL_CARDS-1)
    while deck[card2] <= 0 or card1 == card2:
        card2 = random.randint(0, TOTAL_CARDS-1)
    card3 = random.randint(0, TOTAL_CARDS-1)
    while deck[card3] <= 0 or card1 == card3 or card2 == card3:
        card3 = random.randint(0, TOTAL_CARDS-1)
    deck[card1] -= 1
    deck[card2] -= 1
    deck[card3] -= 1
    return [card1, card2, card3]

def sell_card(card,COST_LOOKUP):
    if card[0] == -1:
        return 0
    return COST_LOOKUP[card[0]]*(2**(card[1]-1)) - 1

def merge_units(arr):
    original_length = len(arr)

    # Remove empty slots
    cards = [c for c in arr if c != (-1, -1)]
    if not cards:
        return [(-1,-1)] * original_length, 0

    total_merges = 0

    while True:
        # Count how many of each (name, level) exist
        counts = {}
        for name, lvl in cards:
            counts.setdefault((name, lvl), 0)
            counts[(name, lvl)] += 1

        merged_any = False
        new_cards = []

        # Process each group
        for (name, lvl), count in counts.items():
            pairs = count // 2
            leftover = count % 2

            # Add merged cards
            for _ in range(pairs):
                new_cards.append((name, lvl + 1))
                total_merges += 1
                merged_any = True

            # Add leftover single
            if leftover == 1:
                new_cards.append((name, lvl))

        cards = new_cards

        # If no merges happened this pass, we are done
        if not merged_any:
            break

    # Pad back to original size
    while len(cards) < original_length:
        cards.append((-1, -1))

    return cards, total_merges

class MERGE_ENV(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):

        
        super(MERGE_ENV, self).__init__()

        self.w_hand = 0.1
        self.w_economy = 0.2
        self.w_merge = 0.05

        self.COST_LOOKUP = {k: v["cost"] for k, v in CARD_TABLE.items()}
        self.NAME_LOOKUP = {k: v["name"] for k, v in CARD_TABLE.items()}

        self.MERGE_ELIXIR = 2

        self.TOTAL_CARDS = 24
        self.TOTAL_COPIES = 4
        
        self.HAND_SLOTS = 9
        self.BUY_SLOTS = 3

        self.MAX_ITR = 200
        self.INVALID_REWARD = -3

        # OBSERVATION = 12-element vector
        self.observation_space = spaces.Box(
            low=-1,
            high=self.TOTAL_CARDS,
            shape=(self.HAND_SLOTS * 2 + self.BUY_SLOTS * 2 + 1,), # 2 For level (Hand) and 2 For Cost (Shop) and 1 For Elixir 
            dtype=np.int32
        )

        # ONLY 0–6 are valid
        self.action_space = spaces.Discrete(self.HAND_SLOTS + self.BUY_SLOTS)

        print("ACTION SPACE SIZE =", self.action_space.n)
        print("MERGE_ENV LOADED FROM:", __file__)

    def reset(self, seed=None, options=None):

        self.itr = 0
 
        self.elixir = 12 # random.randint(15, 20)
        self.starting_elixir = self.elixir

        self.truncated = False
        self.terminated = False
        
        self.deck = [self.TOTAL_COPIES for i in range(self.TOTAL_CARDS)]

        self.hand = [(-1, -1)]*self.HAND_SLOTS

        starting_card = random.randint(0, 6)

        self.hand[0] = (starting_card, 1)

        self.deck[starting_card] -= 1

        self.shop = update_shop(self.deck, self.TOTAL_CARDS)


        # observation : hand and shop (so total of 2*5 + 12 = 7 obseravtions)
        self.observation = []
        for i in range(self.HAND_SLOTS): # Card Name and Level 
            self.observation.append(self.hand[i][0])
            self.observation.append(self.hand[i][1])
        for i in range(self.BUY_SLOTS):
            self.observation.append(self.shop[i])
            self.observation.append(self.COST_LOOKUP[self.shop[i]])
        self.observation.append(self.elixir)
        
        info = {"deck" : self.deck}

        return np.array(self.observation, dtype=np.int32), info

    def step(self, action):
        self.reward = 0

        self.economy_reward = 0
        self.hand_reward = 0
        self.merge_reward = 0

        self.prev_hand = self.hand.copy()
        self.prev_elixir = self.elixir



        hashmap = {
            0: "s0", 1: "s1", 2: "s2", 3: "s3", 4: "s4",
            5: "s5", 6: "s6", 7 : "s7", 8 : "s8", 9 : "b0", 10 : "b1", 11 : "b2"
        }
        
        action = hashmap[int(action)]

        
        if action[0] == "s":
            if self.hand[int(action[1])] == (-1, -1): # Selling Empty Slot 
                self.reward += self.INVALID_REWARD
            else:  
                self.prev_hand = self.hand.copy()
                self.prev_elixir = self.elixir

                self.elixir += sell_card(self.hand[int(action[1])], self.COST_LOOKUP)
                self.deck[self.hand[int(action[1])][0]] += 2**(self.hand[int(action[1])][1]-1)
                self.hand[int(action[1])] = (-1,-1)
                
                self.economy_reward += self.elixir - self.prev_elixir
                self.hand_reward += hand_strength(self.hand, self.COST_LOOKUP) - hand_strength(self.prev_hand, self.COST_LOOKUP)
 
        elif action[0] == "b":
            if (-1, -1) not in self.hand:
                if (self.shop[int(action[1])], 1) in self.hand:
                        if self.elixir < self.COST_LOOKUP[self.shop[int(action[1])]]:
                            self.reward += self.INVALID_REWARD
                        else:
                            self.prev_hand = self.hand.copy()
                            self.prev_elixir = self.elixir

                            self.hand[self.hand.index((self.shop[int(action[1])], 1))] = (self.shop[int(action[1])], 2)
                            self.elixir += self.MERGE_ELIXIR
                            self.hand, self.merges = merge_units(self.hand)
                            self.merges += 1
                            
                            self.elixir += self.merges*self.MERGE_ELIXIR
                            self.elixir -= self.COST_LOOKUP[self.shop[int(action[1])]]
                            for i in range(len(self.shop)):
                                self.deck[self.shop[i]] += 1
                            self.deck[self.shop[int(action[1])]] -= 1
                            self.shop = update_shop(self.deck, self.TOTAL_CARDS)
                            
                            self.merge_reward += self.merges
                            self.economy_reward += self.elixir - self.prev_elixir
                            self.hand_reward += hand_strength(self.hand, self.COST_LOOKUP) - hand_strength(self.prev_hand, self.COST_LOOKUP)
            
                else:
                    self.reward += self.INVALID_REWARD
                    
                    
            else:
                if self.elixir < self.COST_LOOKUP[self.shop[int(action[1])]]:
                    self.reward += self.INVALID_REWARD
                    
                else:
                    self.prev_hand = self.hand.copy()
                    self.prev_elixir = self.elixir

                    self.hand[self.hand.index((-1, -1))] = (self.shop[int(action[1])], 1)
                    self.hand, self.merges = merge_units(self.hand)
                    self.elixir += self.merges*self.MERGE_ELIXIR
                    self.elixir -= self.COST_LOOKUP[self.shop[int(action[1])]]
                    for i in range(len(self.shop)):
                        self.deck[self.shop[i]] += 1
                    self.deck[self.shop[int(action[1])]] -= 1
                    self.shop = update_shop(self.deck, self.TOTAL_CARDS)
                    
                    self.merge_reward += self.merges
                    self.economy_reward += self.elixir - self.prev_elixir
                    self.hand_reward += hand_strength(self.hand, self.COST_LOOKUP) - hand_strength(self.prev_hand, self.COST_LOOKUP)
        
        progress = self.itr/self.MAX_ITR
        
        self.reward += self.w_economy*self.economy_reward*(1 - progress)**2 + self.w_hand*self.hand_reward*((progress)**2) + self.w_merge*self.merge_reward + sum(5 for i in range(len(self.hand)) if self.hand[i][1] == 3) - sum(5 for i in range(len(self.prev_hand)) if self.prev_hand[i][1] == 3)

        self.observation = []
        for i in range(self.HAND_SLOTS): # Card Name and Level 
            self.observation.append(self.hand[i][0])
            self.observation.append(self.hand[i][1])
        for i in range(self.BUY_SLOTS):
            self.observation.append(self.shop[i])
            self.observation.append(self.COST_LOOKUP[self.shop[i]])
        self.observation.append(self.elixir)

        info = {"deck" : self.deck}

        if sum(1 for i in range(len(self.hand)) if self.hand[i] == 3) >= 5:
            self.reward += 100
            self.terminated = True


        self.itr += 1
        self.truncated  = self.itr >= self.MAX_ITR
        return np.array(self.observation, dtype=np.int32), self.reward, self.terminated, self.truncated, info

    def render(self, action):
        print(f"Action : {action}")
        print(f"ELIXIR : {self.elixir}")
        print(f"HAND : {self.hand}")
        print(f"SHOP : {self.shop}")
        print()

        return [self.elixir, self.hand]
    def close(self):
        pass

    def action_masks(self):
        mask = []

        # ---------------- SELL ACTIONS (s0–s9) ----------------
        # Can only sell non-empty hand slots
        for i in range(len(self.hand)):
            mask.append(
                self.hand[i] != (-1, -1)
            )

        # ---------------- BUY ACTIONS (b0–b2) ----------------
        hand_full = (-1, -1) not in self.hand

        for i in range(len(self.shop)):
            card = self.shop[i]

            # No card available
            if card is None:
                mask.append(False)
                continue

            # Not enough elixir to buy
            if self.elixir < self.COST_LOOKUP[card]:
                mask.append(False)
                continue

            # Hand full AND no immediate merge possible
            if hand_full and (card, 1) not in self.hand:
                mask.append(False)
                continue

            mask.append(True)
        mask = np.array(mask, dtype=bool)

        return mask
