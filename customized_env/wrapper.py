import numpy.typing as npt
import gymnasium

from typing import Dict, Tuple
from agents_logic.controllers.simple_controller import Controller
from lux.kit import GameState, obs_to_game_state

from agents_logic.obs2feats.third_stage import SimpleThirdStage2Tensor
from lux.stats import StatsStateDict
from lux.utils import my_turn_to_place_factory
from customized_env.phase_rewards.phase_resolver import PhaseRewarder

from prev_folder.lux_folder.luxai_s2.luxai_s2.env import LuxAI_S2
from prev_folder.lux_folder.luxai_s2.luxai_s2.spaces.act_space import ActionsQueue

from agents_logic.bidding.zero_bid_policy import zero_bid_policy
from agents_logic.factory_placement.random_placement_policy import random_factory_placement


def env(phase: PhaseRewarder, controller: Controller):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(LuxAI_S2(collect_stats=True), phase, controller)
    # This wrapper is only for environments which print results to the terminal
    env = gymnasium.wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = gymnasium.wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = gymnasium.wrappers.OrderEnforcingWrapper(env)
    return env


class LuxOnlyLastStageWrapper(gymnasium.Wrapper):
     def reset(self, **kwargs):
        # we upgrade the reset function here
        self.env: LuxAI_S2
        # we call the original reset function first
        obs = self.env.reset(**kwargs)

        # then use the bid policy to go through the bidding phase
        action = dict()
        for agent in self.env.agents:
            action[agent] = zero_bid_policy(agent, obs)
            # always player_0 - first

        obs, _, _, _ = self.env.step(action)
        obs = {agent: obs_to_game_state(self.env.env_steps, self.env.env_cfg, a_obs)
               for agent, a_obs in obs.items()}

        # while real_env_steps < 0, we are in the factory placement phase
        # so we use the factory placement policy to step through this
        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    obs[agent].teams[agent].place_first,
                    self.env.state.env_steps,
                ):
                    action[agent] = random_factory_placement(agent, obs)
                else:
                    action[agent] = dict()
            obs, _, _, info = self.env.step(action)
            obs = {agent: obs_to_game_state(self.env.env_steps, self.env.env_cfg, a_obs) 
                   for agent, a_obs in obs.items()}
        self.prev_obs = obs

        return obs, info


class LuxControllerWrapper(gymnasium.ActionWrapper):
    def __init__(self, env: LuxOnlyLastStageWrapper, controller: Controller):
        super().__init__(env)
        self._controller = controller
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for agent in obs.keys():
            info[agent] = self._controller.global_action_mask(agent, obs)
        return obs, info

    def step(self, actions: Dict[str, Tuple[npt.NDArray]]):
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        self.env: LuxOnlyLastStageWrapper
        obs, rew, done, info = self.env.step(self.action(actions))
        obs = {agent: obs_to_game_state(self.env.env_steps, self.env.env_cfg, a_obs) 
               for agent, a_obs in obs.items()}
        for agent in obs.keys():
            info[agent] = self._controller.global_action_mask(agent, obs)
        return obs, rew, done, done, info

    def action(self, actions: Dict[str, Tuple[npt.NDArray]]) -> Dict[str, ActionsQueue]:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        lux_actions = dict()
        self.env: LuxOnlyLastStageWrapper
        obs = self.env.env.state.get_obs()
        obs = obs_to_game_state(self.env.env.env_steps, self.env.env.env_cfg, obs)

        for agent_name, actions_tuple in actions.items():
            lux_actions[agent_name] = self._controller.action_to_lux_action(
                agent_name, 
                {agent_name: obs}, 
                actions_tuple
            )

        return lux_actions

class LuxRewardWrapper(gymnasium.RewardWrapper):
    def __init__(self, env: LuxControllerWrapper, phase_rewarder: PhaseRewarder):
        super().__init__(env)
        self._phase_rewarder = phase_rewarder

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.env: LuxControllerWrapper
        stats: Dict[str, StatsStateDict] = self.env.env.env.state.stats

        reward = self._phase_rewarder.estimate_reward(
            stats=stats,
            obs=obs,
            done=terminated
        )

        return obs, reward, terminated, truncated, info

    def reward(self, reward: Dict[str, float]):
        pass


class LuxObsWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env: LuxRewardWrapper):
        super().__init__(env)
        self._obs_proc = SimpleThirdStage2Tensor()

    def get_stats(self):
        self.env: LuxRewardWrapper
        return self.env.env.env.env.state.stats

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        old_obs = obs
        obs = self.observation(obs)
        for agent_name in obs.keys():
            obs[agent_name].update(info[agent_name])
        return obs, old_obs

    def step(self, actions: Dict[str, Tuple[npt.NDArray]]):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        info["obs"] = obs
        obs = self.observation(obs)
        for agent_name in obs.keys():
            obs[agent_name].update(info[agent_name])
        return obs, reward, terminated, truncated, info

    def observation(self, observation: Dict[str, GameState]):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """

        obs: Dict[str, Dict[str, npt.NDArray]] = dict()
        for agent, agent_obs in observation.items():
            obs[agent] = self._obs_proc(agent, agent_obs)
        return obs

def raw_env(
    env: LuxAI_S2,
    controller: Controller,
    phase_rewarder: PhaseRewarder,
    feature_observer = None
) -> LuxObsWrapper:
    env = LuxOnlyLastStageWrapper(env)
    env = LuxControllerWrapper(env, controller)
    env = LuxRewardWrapper(env, phase_rewarder)
    env = LuxObsWrapper(env)
    return env

gymnasium.register(
    id="PhasedLuxAI_S2-v0",
    entry_point="customized_env.wrapper:PhasedLuxAI_S2"
)
