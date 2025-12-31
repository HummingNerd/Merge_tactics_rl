from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from env import MERGE_ENV
from matplotlib import pyplot

# Elixir Behaviour 
# Hand Level Behaviour 

def hand_level(hand):
    ans = 0
    for i in range(len(hand)):
        if hand[i][1] == -1:
            continue
        ans += hand[i][1]
    return ans

def elixir_in_hand(hand, card2cost):
    ans = 0
    for i in range(len(hand)):
        if hand[i][1] == -1:
            continue
        ans += card2cost[hand[i][0]]*(2**(hand[i][1] - 1)) - 1
    return ans



def test_trained_model(model_path, episodes=3, max_steps=500):
    print(f"\n🎯 Loading trained model from: {model_path}")

    model = MaskablePPO.load(model_path)

    env = MERGE_ENV()

    fig, (ax1, ax2) = pyplot.subplots(1, 2) 

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Elixir")
    ax1.set_title("Elixir over Time")

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Hand Level")
    ax2.set_title("Hand Level over Time")

    best_hand = []

    for ep in range(episodes):
        elixir_history = []
        level_history = []
        obs, info = env.reset()
        terminated, truncated = False, False
        step = 0

        sub_best_hand = []
        max_level = 0


        print("\n" + "=" * 40)
        print(f"▶️  EPISODE {ep + 1}")
        print("=" * 40)

        while not (terminated or truncated) and step < max_steps:
            step += 1

            # 🔑 GET ACTION MASKS FROM ENV
            action_masks = get_action_masks(env)

            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True
            )

            print(f"\nStep {step}")
            print(f"Action taken: {action}")

            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Reward: {reward}")
            print(f"Elixir : ", env.elixir)
            print(f"Deck : {info["deck"]}")
            print("Hand:", env.hand)
            print("Shop:", env.shop)
            print("Observation : ", obs)

            elixir_history.append(
                env.elixir # + elixir_in_hand(env.hand, env.COST_LOOKUP)
            )
            level_history.append(
                hand_level(env.hand)
            )
            if hand_level(env.hand) > max_level:
                sub_best_hand = env.hand
                max_level = hand_level(env.hand)


            
            print()
        

        print(f"\n🏁 Episode {ep + 1} finished after {step} steps")
        best_hand.append(sub_best_hand)

        steps = list(range(len(elixir_history)))

        ax1.plot(steps, elixir_history, label=f"Episode {ep+1}")
        ax2.plot(steps, level_history, label=f"Episode {ep+1}")
    
    ax1.legend()
    ax2.legend()
    for hand in best_hand:
        print(hand)
    pyplot.show()

    env.close() 


if __name__ == "__main__":
    model_path = "models/merge_env/ppo_merge_final.zip" 
    model_path = "models/merge_env/best/best_model.zip"
    test_trained_model(model_path, episodes=100, max_steps=200)
    
