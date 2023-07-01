import sys
import torch

from telnetlib import GA
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import numpy.typing as npt
from gym import spaces
from pyparsing import Optional
from lux.factory import Factory

from lux.kit import Board, GameState
from lux.unit import Unit, move_deltas


ActionType = npt.NDArray[np.int_]

# Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server
class Controller:
    def __init__(self):
        self.water_act_dims = 1
        self.light_build_dims = 1
        self.heavy_build_dims = 1
        self.fact_no_op_dims = 1

        self.water_dim = 2
        self.light_dim = 0
        self.heavy_dim = 1
        self.fact_no_op_dim = 3

        self.unit_total_act_dims = 1
        self.factory_total_act_dims = 4

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ) -> Dict[str, List[ActionType]]:
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def global_action_mask(self, agent: str, obs: Dict[str, GameState]) -> Dict[str, npt.NDArray]:
        unit_masks = self.robots_action_masks(agent, obs)
        state = obs[agent]
        max_H, max_W = state.board.rubble.shape[:2]
        global_unit_mask = np.zeros((max_H, max_W, self.unit_total_act_dims), dtype=np.bool8)
        global_fact_mask = np.zeros((max_H, max_W, self.factory_total_act_dims), dtype=np.bool8)

        for unit_name, unit_mask in unit_masks.items():
            unit_pos = state.units[agent][unit_name].pos
            x, y = unit_pos[:2]
            global_unit_mask[x, y, :] = unit_mask

        fact_masks = self.factory_action_masks(agent, obs)
        for fact_name, fact_mask in fact_masks.items():
            fact_pos = state.factories[agent][fact_name].pos
            x, y = fact_pos[:2]
            global_fact_mask[x, y, :] = fact_mask

        return {
            "unit_mask": global_unit_mask.transpose(2, 0, 1),
            "factory_mask": global_fact_mask.transpose(2, 0, 1)
        }

    def action_masks(self, agent: str, obs: Dict[str, GameState]) -> Dict[str, npt.NDArray]:
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        actions_masks = self.factory_action_masks(agent, obs)
        actions_masks.update(self.robots_action_masks(agent, obs))
        return actions_masks

    def robots_action_masks(self, agent: str, obs: Dict[str, GameState]) -> Dict[str, npt.NDArray]:
        raise NotImplementedError()

    def factory_action_masks(self, agent: str, obs: Dict[str, GameState]) -> Dict[str, npt.NDArray]:
        actions_masks: Dict[str, npt.NDArray] = dict()
        state = obs[agent]
        for factory_name, factory_obj in state.factories[agent].items():
            factory_obj: Factory
            valid_mask = np.zeros((self.factory_total_act_dims), dtype=np.bool8)
            valid_mask[self.water_dim] = factory_obj.can_water(state)
            valid_mask[self.light_dim] = factory_obj.can_build_light(state)
            valid_mask[self.heavy_dim] = factory_obj.can_build_heavy(state)
            valid_mask[self.fact_no_op_dim] = np.True_
            actions_masks[factory_name] = valid_mask

        return actions_masks


class SimpleUnitDiscreteController(Controller):
    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.

        For the robot unit:
        - 4 cardinal direction movement (4 dims)
        - transfer action just for transferring ice/ore in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power
        - recharge action - to full (1 dim)
        Total - 13 actions

        For the factory:
        - water-action
        - build light
        - build heavy
        - no op
        Total - 4 actions


        Does not include
        - transferring power, water, metal
        - pickup water, ice, ore, metal
        - self-desctruction
        - move center(useful for planning with action-queue)

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        super().__init__()
        self.env_cfg = env_cfg

        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1
        self.recharge_act_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims
        self.recharge_dim_high = self.no_op_dim_high + self.recharge_act_dims

        self.unit_total_act_dims = self.recharge_dim_high

        self.movements = list(range(1, 5))
        self.transfer_movements = list(range(0, 5))

    def _is_no_op(self, id):
        return id + 1 == self.no_op_dim_high

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id, resouce_id: int):
        # 0 == ice
        # 1 == ore
        id = id - self.move_dim_high
        transfer_dir = id
        return np.array([1, transfer_dir, resouce_id, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def _get_recharge_action(self, id, unit: Unit):
        return np.array([5, 0, 0, unit.get_max_battery(), 0, 1])

    def unit_action_idx_to_lux(self, unit: Unit, choice: int, state: GameState) -> List[ActionType]:
        if self._is_move_action(choice):
            return [self._get_move_action(choice)]
        if self._is_transfer_action(choice):
            queue = []
            if unit.cargo.ice > 0:
                queue.append(self._get_transfer_action(choice, 0))
            if unit.cargo.ore > 0:
                queue.append(self._get_transfer_action(choice, 1))
            return queue
        if self._is_pickup_action(choice):
            return [self._get_pickup_action(choice)]
        if self._is_dig_action(choice):
            return [self._get_dig_action(choice)]
        return [self._get_recharge_action(choice, unit)]

    def make_choice(self, pos: npt.NDArray, mask: npt.NDArray,
                    probs: npt.NDArray, act_idx: npt.NDArray) -> int:
        x, y = pos[:2]
        probs = probs[x, y]
        act_idx = act_idx[mask]
        probs = probs[mask]
        return np.random.choice(act_idx, probs)

    def action_to_lux_action(
        self,
        agent: str,
        obs: Dict[str, GameState],
        action: Tuple[npt.NDArray, npt.NDArray]
    ) -> Dict[str, List[ActionType]]:
        lux_action: Dict[str, List[ActionType]] = dict()
        (unit_actions, fact_actions) = action
        # action_masks = self.action_masks(agent, obs)
        state: GameState = obs[agent]
        # actions_idx = np.arange(self.unit_total_act_dims)
        for unit_name, unit in state.units[agent].items():
            unit: Unit
            # mask = action_masks[unit_name]
            # choice = self.make_choice(unit.pos, mask, unit_actions, actions_idx)
            x, y = unit.pos[:2]
            choice = unit_actions[x, y]
            if not self._is_no_op(choice):
                lux_action[unit_name] = self.unit_action_idx_to_lux(unit, choice, state)
        for fact_name, fact in state.factories[agent].items():
            fact: Factory
            # mask = action_masks[fact_name]
            # choice = self.make_choice(unit.pos, mask, fact_actions, actions_idx)
            x, y = fact.pos[:2]
            choice = fact_actions[x, y]
            if choice != 3: # no op for fact
                lux_action[fact_name] = choice
        return lux_action

    def robots_action_masks(self, agent: str, obs: Dict[str, GameState]) -> Dict[str, npt.NDArray]:
        """
        Defines a simplified action mask for this controller's action space
        """
        actions_masks: Dict[str, npt.NDArray] = dict()
        state = obs[agent]
        for unit_name, unit_obj in state.units[agent].items():
            unit_obj: Unit
            valid_mask = np.zeros(self.unit_total_act_dims, dtype=np.bool8)
            valid_mask[:self.move_dim_high] = self.move_action(unit_obj, state)
            valid_mask[self.move_dim_high:self.transfer_dim_high] = self.transert_action(unit_obj, state)
            valid_mask[self.transfer_dim_high:self.pickup_dim_high] = self.pickup_action(unit_obj, state)
            valid_mask[self.pickup_dim_high:self.dig_dim_high] = self.dig_action(unit_obj, state)
            valid_mask[self.dig_dim_high:self.no_op_dim_high] = np.ones((self.no_op_dims,), dtype=np.bool8)
            valid_mask[self.no_op_dim_high:self.recharge_dim_high] = self.recharge_action(unit_obj, state)
            actions_masks[unit_name] = valid_mask

        return actions_masks

    def move_action(self, unit: Unit, state: GameState) -> Union[npt.NDArray, List[bool]]:
        mask = np.zeros((self.move_act_dims,), dtype=bool)
        for idx, dir in enumerate(self.movements):
            power_cost: Optional[int] = unit.move_cost(state, dir)
            if power_cost is None:
                continue
            power_cost += unit.action_queue_cost(state)
            if power_cost >= unit.power:
                mask[idx] = np.True_
        return mask

    def transert_action(self, unit: Unit, state: GameState) -> Union[npt.NDArray, List[bool]]:
        mask = np.zeros((self.transfer_act_dims,), dtype=bool)
        if unit.cargo.ice == 0 and unit.cargo.ore == 0:
            return mask
        for idx, dir in enumerate(self.movements):
            to_freind_fact = unit.check_transfer_location(state, dir)
            if not to_freind_fact:
                continue
            power_cost: int = unit.action_queue_cost(state)
            if power_cost >= unit.power:
                mask[idx] = np.True_
        return mask

    def pickup_action(self, unit: Unit, state: GameState) -> Union[npt.NDArray, List[bool]]:
        mask = np.zeros((self.pickup_act_dims,), dtype=bool)
        if unit.power == unit.get_max_battery():
            return mask
        from_freind_fact = unit.check_transfer_location(state, 0)
        if not from_freind_fact:
            return mask
        power_cost: int = unit.action_queue_cost(state)
        if power_cost >= unit.power:
            mask[0] = np.True_
        return mask

    def dig_action(self, unit: Unit, state: GameState) -> Union[npt.NDArray, List[bool]]:
        mask = np.zeros((self.dig_act_dims,), dtype=bool)
        board = state.board
        x, y = unit.pos[:2]
        if board.rubble[x, y] > 0 or board.ice[x, y] > 0 \
            or board.ore[x, y] > 0 or board.lichen[x, y] > 0:
            power_cost: int = unit.action_queue_cost(state) + unit.dig_cost(state)
            if power_cost >= unit.power:
                mask[0] = True
        return mask

    def recharge_action(self, unit: Unit, state: GameState) -> Union[npt.NDArray, List[bool]]:
        mask = np.zeros((self.dig_act_dims,), dtype=bool)
        if unit.power == unit.get_max_battery():
            return mask
        power_cost: int = unit.action_queue_cost(state)
        if power_cost >= unit.power:
            mask[0] = True
        return mask
