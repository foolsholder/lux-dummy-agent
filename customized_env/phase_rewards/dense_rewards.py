from lib2to3.pgen2.token import OP
from typing import Dict, Optional, Any

import numpy as np
from numpy.typing import NDArray
from copy import deepcopy

from lux.stats import *
from lux.kit import *
from customized_env.phase_rewards.phase_resolver import PhaseRewarder


class DenseRewards(PhaseRewarder):
    def __init__(self, reward_dict: Dict[str, float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_init = False
        self.reward_dict: Dict[str, Any] = reward_dict
        self.prev_stats: Optional[StatsStateDict] = None

    def calculate_lichen(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> int:
        return stats[agent]["lichen"]

    def get_lichen_diff(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> int:
        lichen_sum = 0
        for stats_agent_dict in stats.values():
            lichen_sum += stats_agent_dict["lichen"]
        return 2 * self.calculate_lichen(stats, agent, obs) - lichen_sum

    def calculate_transfer_to_factory_reward(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> float:
        reward = 0
        agent_stats = stats[agent]
        prev_stats = self.prev_stats[agent]
        for resource in ["ice", "ore"]:
            reward += self.reward_dict["transfer"][resource] * (
                agent_stats["transfer"][resource] - prev_stats["transfer"][resource]
            )
        return reward

    def calculate_dig_reward(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> float:
        reward = 0
        agent_stats = stats[agent]
        prev_stats = self.prev_stats[agent]
        for resource in ["ice", "ore"]:
            count = 0
            substats = agent_stats["generation"][resource]
            for robot_type in ["LIGHT", "HEAVY"]:
                count += substats[robot_type] - prev_stats["generation"][resource][robot_type]
            reward += self.reward_dict["dig"][resource] * count
        for robot_type in ["LIGHT", "HEAVY"]:
            reward += self.reward_dict["dig"]["rubble"] * (
                agent_stats["destroyed"]["rubble"][robot_type] - prev_stats["destroyed"]["rubble"][robot_type]
            )
        return reward

    def calculate_pickup_power_reward(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> float:
        reward = 0
        agent_stats = stats[agent]
        prev_stats = self.prev_stats[agent]
        for robot_type in ["LIGHT", "HEAVY"]:
            reward += self.reward_dict["pickup_power"][robot_type] * (
                agent_stats["pickup"]["power"] - prev_stats["pickup"]["power"]
            )
        return reward

    def calculate_penalty(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> float:
        agent_stats = stats[agent]
        prev_stats = self.prev_stats[agent]
        penalty = 0
        for type in ["LIGHT", "HEAVY", "FACTORY"]:
            penalty += self.reward_dict["penalty"][type] * (
                agent_stats["destroyed"][type] - prev_stats["destroyed"][type]
            )
        if len(obs.factories[agent]) == 0:
            penalty += self.reward_dict["penalty"]["all_factories"]
        return penalty

    def calculate_facrory_reward(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> float:
        agent_stats = stats[agent]
        prev_stats = self.prev_stats[agent]
        reward = 0
        for robot_type in ["LIGHT", "HEAVY"]:
            reward += self.reward_dict["creation"][robot_type] * (
                agent_stats["generation"]["built"][robot_type] - prev_stats["generation"]["built"][robot_type]
            )
        return reward

    def calculate_movement_reward(
        self,
        stats: Dict[str, StatsStateDict],
        agent: str, obs: GameState
    ) -> float:
        agent_stats = stats[agent]
        prev_stats = self.prev_stats[agent]
        reward = self.reward_dict["movement"] * (
            agent_stats["movement_success"] - prev_stats["movement_success"]
        )
        return reward

    def calculate_reward(
        self,
        stats: Dict[str, StatsStateDict],
        obs: GameState, done: bool, agent: str
    ) -> float:
        reward = 0
        if self.is_init:
            reward += self.calculate_facrory_reward(stats, agent, obs)
            reward += self.calculate_transfer_to_factory_reward(stats, agent, obs)
            reward += self.calculate_movement_reward(stats, agent, obs)
            reward += self.calculate_dig_reward(stats, agent, obs)
            reward += self.calculate_pickup_power_reward(stats, agent, obs)
            reward -= self.calculate_penalty(stats, agent, obs)
        if done:
            reward += self.reward_dict["win_coef"] * np.sqrt(
                max(0, self.get_lichen_diff(stats, agent, obs))
            )
        return reward

    def estimate_reward(
        self,
        stats: Dict[str, StatsStateDict],
        obs: Dict[str, GameState],
        done: Dict[str, bool]
    ) -> Dict[str, float]:
        reward = dict()
        for agent_name in stats.keys():
            reward[agent_name] = self.calculate_reward(
                stats, obs[agent_name], done[agent_name], agent_name
            )
        self.is_init = True
        self.prev_stats = deepcopy(stats)
        return reward
