from gym.envs.registration import register

register(
    id='reversi-v0',
    entry_point='gym_reversi.envs:ReversiEnv',
)
