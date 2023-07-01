from typing import Dict, List, Union, Tuple
from more_itertools import sample
import torch
from torch.utils.data import Dataset

import numpy as np
import numpy.typing as npt
import random


class RolloutBuffer:
    def __init__(self) -> None:
        self.trajectory = []
        self.critic_values = []
        self.advantages = []
        self.returns = []

    def __len__(self):
        return len(self.trajectory)

    def move_data_from(self, buff: "RolloutBuffer"):
        self.add_trajectory(
            buff.trajectory,
            buff.critic_values,
            buff.advantages,
            buff.returns
        )
        buff.reset_buffer()

    def reset_buffer(self):
        self.trajectory = []
        self.critic_values = []
        self.advantages = []
        self.returns = []

    def add_trajectory(self, trajectory, critic_values, advantages, returns):
        self.trajectory.extend(trajectory)
        self.critic_values.extend(critic_values)
        self.advantages.extend(advantages)
        self.returns.extend(returns)

    def shuffle_truncate(self, buffer_size: int, shuffle_seed: int):
        if len(self.trajectory) <= buffer_size:
            return

        random.Random(shuffle_seed).shuffle(self.trajectory)
        random.Random(shuffle_seed).shuffle(self.critic_values)
        random.Random(shuffle_seed).shuffle(self.advantages)
        random.Random(shuffle_seed).shuffle(self.returns)

        del self.trajectory[buffer_size:]
        del self.critic_values[buffer_size:]
        del self.advantages[buffer_size:]
        del self.returns[buffer_size:]


DATASET_SAMPLE_TYPE = Dict[str, Union[
        Dict[str, npt.NDArray], npt.NDArray, Tuple[npt.NDArray, npt.NDArray]
    ]
]

def proc_trajectory_to_dataset(
    traj, values, adv, returns
) -> List[DATASET_SAMPLE_TYPE]:
    res: List[DATASET_SAMPLE_TYPE] = []
    obs_dict, actions_dict, reward_dict, next_obs_dict, done_flag = traj
    for agent_name in reward_dict:
        sample_dict: Dict[str, npt.NDArray] = dict()
        sample_dict['critic_target'] = returns[agent_name]
        sample_dict['critic_old'] = values[agent_name]
        sample_dict['observation'] = obs_dict[agent_name] # Dict[str, npt.NDArray]
        sample_dict['action'] = actions_dict[agent_name]['action_sample'] # Tuple[npt.NDArray]
        # print(len(sample_dict['action'] ), type(sample_dict['action'] ))
        sample_dict['action_logprob'] = actions_dict[agent_name]['action_logprob'] # Tuple[]
        sample_dict['gae_advantage'] = adv[agent_name]
        res.append(sample_dict)
    return res


class TrajectoryDataset(Dataset):
    def __init__(self, rollout_buffer: RolloutBuffer) -> None:
        super().__init__()
        self.self_play_trajectory = []

        traj, values, adv, returns = (
            rollout_buffer.trajectory,
            rollout_buffer.critic_values,
            rollout_buffer.advantages,
            rollout_buffer.returns
        )
        # print(len(traj))
        for t, v, a, r in zip(traj, values, adv, returns):
            self.self_play_trajectory.extend(
                proc_trajectory_to_dataset(t, v, a, r)
            )

    def __len__(self) -> int:
        return len(self.self_play_trajectory)

    def __getitem__(
        self,
        index: int
    ) -> DATASET_SAMPLE_TYPE:
        return self.self_play_trajectory[index]
