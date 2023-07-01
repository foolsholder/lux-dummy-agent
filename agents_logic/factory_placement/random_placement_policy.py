import numpy as np

from typing import Union, Dict

from lux.kit import GameState
from lux.utils import my_turn_to_place_factory

def random_factory_placement(agent: str, obs: Dict[str, GameState]):
    # factory placement period
    state = obs[agent]
    # how much water and metal you have in your starting pool to give to new factories
    # water_left = state.teams[agent].water
    # metal_left = state.teams[agent].metal

    # how many factories you have left to place
    factories_to_place = state.teams[agent].factories_to_place
    # whether it is your turn to place a factory
    my_turn_to_place = my_turn_to_place_factory(state.teams[agent].place_first, state.env_steps)
    if factories_to_place > 0 and my_turn_to_place:
        # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
        potential_spawns = np.array(list(zip(*np.where(state.board.valid_spawns_mask == 1))))
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        return dict(spawn=spawn_loc, metal=150, water=150)
    return dict()
