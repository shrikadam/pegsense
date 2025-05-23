from gymnasium.envs.registration import register

register(
    id="ur10e_mujoco_env/UR10eMjEnv-v0",
    entry_point="ur10e_mujoco_env.env:UR10eMjEnv",
)