import gymnasium as gym
from env import UR5eMjEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# 1. Create Env
env = UR5eMjEnv(render_mode=None) # Set "human" to watch, None to train fast

# 2. Sanity Check (Catches shape mismatches early)
print("Checking environment...")
check_env(env)
print("Environment is valid!")

# 3. Initialize Agent
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# 4. Train (Short run to prove it runs)
print("Starting training...")
model.learn(total_timesteps=10000)

# 5. Visualize Result
print("Training finished. Visualizing...")
env = UR5eMjEnv(render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, _ = env.reset()