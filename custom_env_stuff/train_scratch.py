import os
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from env import MERGE_ENV


# =========================================================
# Paths
# =========================================================
LOG_DIR = "logs/merge_env/"
SAVE_DIR = "models/merge_env/"
BEST_DIR = os.path.join(SAVE_DIR, "best")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)


# =========================================================
# Action mask
# =========================================================
def mask_fn(env):
    return env.action_masks()


# =========================================================
# Make env (SINGLE ENV ONLY)
# =========================================================
def make_env():
    env = MERGE_ENV()
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    return env


# =========================================================
# Hand strength (true objective)
# =========================================================
def hand_strength(hand, card2cost):
    strength = 0.0
    for card_id, lvl in hand:
        if card_id == -1:
            continue
        strength += np.sqrt(card2cost[card_id]) * (2 ** (lvl - 1))
    return strength


# =========================================================
# Evaluation (NO VecEnv BS)
# =========================================================
def evaluate_hand_strength(model, env, episodes=10, max_steps=200):
    scores = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            masks = env.env.action_masks()   # ✅ FIX
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=masks
            )

            obs, _, done, _, _ = env.step(action)
            steps += 1

        raw_env = env.env.env              # ✅ MERGE_ENV
        scores.append(
            hand_strength(raw_env.hand, raw_env.COST_LOOKUP)
        )

    return float(np.mean(scores))


# =========================================================
# Callback: save best model
# =========================================================
class BestHandStrengthCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq=100_000,
        n_eval_episodes=10,
        max_eval_steps=200,
        verbose=1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_eval_steps = max_eval_steps
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            score = evaluate_hand_strength(
                self.model,
                self.eval_env,
                self.n_eval_episodes,
                self.max_eval_steps
            )

            print(
                f"[Eval @ {self.num_timesteps}] "
                f"Hand strength: {score:.2f} "
                f"(best: {self.best_score:.2f})"
            )

            if score > self.best_score:
                self.best_score = score
                self.model.save(os.path.join(BEST_DIR, "best_model"))
                print("✅ New best model saved")

        return True


# =========================================================
# Environments
# =========================================================
train_env = make_env()
eval_env = make_env()


# =========================================================
# PPO (SAFE hyperparameters for your env)
# =========================================================
model = MaskablePPO(
    "MlpPolicy",
    train_env,

    ent_coef=0.005,
    learning_rate=1e-4,

    n_steps=256,
    batch_size=128,

    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,

    verbose=1,
    tensorboard_log=LOG_DIR,
)


# =========================================================
# Train
# =========================================================
callback = BestHandStrengthCallback(
    eval_env=eval_env,
    eval_freq=100_000
)

model.learn(
    total_timesteps=5_000_000,
    callback=callback
)

model.save(os.path.join(SAVE_DIR, "ppo_merge_final"))


# =========================================================
# Final evaluation
# =========================================================
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=50,
    deterministic=True
)

print(f"\n✅ Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")