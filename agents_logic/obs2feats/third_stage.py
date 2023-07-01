import gymnasium
import torch
import numpy as np

from numpy.typing import NDArray
from lux.kit import *

POWER_INDEX = 4
ICE_INDEX = 0
ORE_INDEX = 1
WATER_INDEX = 2
METAL_INDEX = 3

class SimpleThirdStage2Tensor:
    def gather_unit_info(self, obs: GameState, agent: str, unit_type: str) -> torch.Tensor:
        war_mask = np.zeros_like(obs.board.rubble)
        unit_info = np.zeros((3,) + tuple(war_mask.shape))
        for agent_name, unit_dict in obs.units.items():
            war_value = 1 if agent == agent_name else -1
            for unit in unit_dict.values():
                unit: Unit
                if unit.unit_type != unit_type:
                    continue
                x, y = unit.pos[0], unit.pos[1]
                war_mask[x, y] = war_value

                unit_info[0, x, y] = unit.power
                cargo = unit.cargo
                unit_info[ICE_INDEX, x, y] = cargo.ice
                unit_info[ORE_INDEX, x, y] = cargo.ore
        return torch.cat([
            torch.from_numpy(unit_info), torch.from_numpy(war_mask[None])
        ], dim=0).float() # [4, H, W]

    def create_lichen_mask(self, obs: GameState, agent: str) -> NDArray:
        strain2war_value = dict()
        max_index = 0
        for agent_name, team in obs.teams.items():
            war_value = 1 if agent == agent_name else -1
            factory_strains = team.factory_strains
            for strain in factory_strains:
                strain2war_value[strain] = war_value
                max_index = max(max_index, strain)

        strain2war_value_list = [-1] * (max_index + 1)
        for strain, war_value in strain2war_value.items():
            strain2war_value_list[strain] = war_value
        strain2war_value_list = np.array(strain2war_value_list, dtype=int)
        lichen_strains: NDArray = obs.board.lichen_strains
        shape = lichen_strains.shape
        lichen_strains = lichen_strains.reshape(-1)
        lichen_team = np.where(
            lichen_strains != -1,
            strain2war_value_list[lichen_strains],
            0
        )
        return lichen_team.reshape(shape)

    def board2tensor(self, obs: GameState, agent: str) -> torch.Tensor:
        board = obs.board
        rubble: NDArray = board.rubble
        ice: NDArray = board.ice
        ore: NDArray = board.ore
        lichen: NDArray = board.lichen
        lichen_mask: NDArray = self.create_lichen_mask(obs, agent)
        return torch.from_numpy(
            np.stack(
                [
                    rubble,
                    ice,
                    ore,
                    lichen,
                    lichen_mask
                ],
                axis=0,
            ),
        ).float() # [5, H, W]

    def units2tensor(self, obs: GameState, agent: str) -> torch.Tensor:
        return torch.cat([
            self.gather_unit_info(obs, agent, "LIGHT"), self.gather_unit_info(obs, agent, "HEAVY")
        ], dim=0) # [8, H, W]

    def factories2tensor(self, obs: GameState, agent: str) -> torch.Tensor:
        war_mask = np.zeros_like(obs.board.rubble, dtype=np.float32)
        factory_info = np.zeros((5,) + tuple(war_mask.shape), dtype=np.float32)
        for agent_name, factory_dict in obs.factories.items():
            war_value = 1 if agent == agent_name else -1
            for factory in factory_dict.values():
                factory: Factory
                x, y = factory.pos[0], factory.pos[0]

                war_mask[x, y] = war_value
                factory_info[POWER_INDEX, x, y] = factory.power
                cargo = factory.cargo
                factory_info[ICE_INDEX, x, y] = cargo.ice
                factory_info[ORE_INDEX, x, y] = cargo.ore
                factory_info[WATER_INDEX, x, y] = cargo.water
                factory_info[METAL_INDEX, x, y] = cargo.metal

        return torch.cat([
            torch.from_numpy(factory_info), torch.from_numpy(war_mask[None])
        ], dim=0).float() # [6, H, W]

    def actions2tensor(self, obs: GameState, agent: str) -> torch.Tensor:
        return torch.zeros(1)

    def __call__(self, agent: str, obs: GameState) -> Dict[str, torch.Tensor]:
        board_tensor = self.board2tensor(obs, agent)
        units_tensor = self.units2tensor(obs, agent)
        factories_tensor = self.factories2tensor(obs, agent)

        map_info = torch.cat([
            board_tensor,
            units_tensor,
            factories_tensor,
        ], dim=0).float() # [19, H, W]

        # actions_info = self.actions2tensor(obs, agent)
        real_env_steps = torch.Tensor([obs.real_env_steps]).float()

        return {
            "map_info": map_info,
            # "actions_info": actions_info,
            "global_time": real_env_steps
        }