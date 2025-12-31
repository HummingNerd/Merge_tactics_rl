from env import MERGE_ENV
def test_env(env, steps=10):
    print("🔍 Testing environment...")

    obs, info = env.reset()
    print("Initial observation:", obs)
    i = 0
    while i < steps:
        action = env.action_space.sample()
        
        if env.action_masks()[action] == False:
            continue

        print(f"\n➡️ Step {i}, Action = {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print()
        print("Reward: ", reward)
        print("Observation:", obs)
        print("Terminated:", terminated)
        print("Truncated:", truncated)


        if terminated or truncated:
            print("Environment ended, resetting...\n")
            obs, info = env.reset()
        i += 1

    print("✅ Environment test complete!")

env = MERGE_ENV()
test_env(env, steps = 10)