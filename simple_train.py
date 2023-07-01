from ast import arg
from git import Object
from customized_env.wrapper import LuxObsWrapper, raw_env, env as wrapper_env
from customized_env.cleanrl.ppo.ppo import PPO

from agents_logic.neural_net.simple_net import SimpleCentralizedNet
from agents_logic.controllers.simple_controller import Controller, SimpleUnitDiscreteController

from customized_env.phase_rewards.dense_rewards import DenseRewards

from luxai_s2.env import LuxAI_S2

from omegaconf import DictConfig, OmegaConf
import hydra
import omegaconf

import torch
from torch import distributed as dist
import os

def init_process(cfg, local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    fn(cfg, local_rank)

def main_impl(cfg: DictConfig, rank: int):
    if rank == 0:
        print(OmegaConf.to_yaml(cfg))
        print(OmegaConf.to_yaml(cfg['algorithm']))
    policy_network = SimpleCentralizedNet(cfg['agent_net'])
    rewarder = DenseRewards(cfg['rewards'])

    raw_lux = LuxAI_S2(collect_stats=True)
    controller = SimpleUnitDiscreteController(raw_lux.env_cfg)
    env = raw_env(raw_lux, controller, rewarder)

    device = torch.device('cuda', rank)

    ppo_learner = PPO(
        policy_network,
        cfg['algorithm'],
        device=device
    )

    batch_size = 512
    ppo_learner.learn(
        n_ppo_iters=100,
        env=env,
        n_buffer_size=batch_size*30,
        max_env_steps=300,
        ckpt_folder="checkpoints/simple",
        n_train_epochs=4,
        batch_size=batch_size,
        num_workers=5,
        global_train_cfg=cfg
    )

@hydra.main(config_path="configs", config_name="simple.yaml")
def main(cfg: DictConfig):
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(cfg, local_rank, fn=main_impl, backend="nccl")

if __name__ == '__main__':
    main()