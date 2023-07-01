from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import copy

from torch import nn, distributed as dist
from torch.utils.data.distributed import DistributedSampler

import os.path as osp

import numpy as np
import numpy.typing as npt
from tqdm.auto import trange

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb

from customized_env.cleanrl.ppo.trajectory_buffer import RolloutBuffer, TrajectoryDataset
from customized_env.wrapper import LuxObsWrapper

from agents_logic.neural_net.simple_net import SimpleCentralizedNet

@torch.no_grad()
def get_w_grad_norm(model):
    w_norm = torch.sqrt(sum([torch.sum(t.data ** 2) for t in model.parameters()]))
    grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in model.parameters()]))
    return w_norm, grad_norm

def dict_to_device(obs_dict: Dict[str, Dict[str, Union[torch.Tensor, npt.NDArray]]], device = 'cpu'):
    tensor_dict: Dict[str, Dict[str, torch.Tensor]] = dict()
    for agent_name, dct in obs_dict.items():
        tensor_dict[agent_name] = {}
        for key, value in dct.items():
            if isinstance(value, torch.Tensor):
                tensor_dict[agent_name][key] = value.to(device)[None]
            else:
                tensor_dict[agent_name][key] = torch.from_numpy(value).to(device)[None]
            # print(key, tensor_dict[agent_name][key].shape)
    return tensor_dict

def tensor_out_to_numpy(tensor_out: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor]]]):
    out: Dict[str, npt.NDArray] = dict()
    for key, value in tensor_out.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.cpu().numpy()[0]
        else:
            out[key] = tuple((vv.cpu().numpy()[0] for vv in value))
    return out

DATALOADER_SAMPLE_TYPE = Dict[str, Union[
        Dict[str, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]
]

def tensor_dict_to_device(dct: DATALOADER_SAMPLE_TYPE, device = 'cpu') -> DATALOADER_SAMPLE_TYPE:
    res = dict()
    for key, value in dct.items():
        # print(key, "type", type(value))
        if isinstance(value, torch.Tensor):
            res[key] = value.to(device)
        elif isinstance(value, tuple) or isinstance(value, list):
            res[key] = tuple((vv.to(device) for vv in value))
        elif isinstance(value, dict):
            res[key] = tensor_dict_to_device(value, device)
        else:
            raise "What's in your dict, bro?"
    return res

class PPO:
    def __init__(
        self,
        policy_net: SimpleCentralizedNet,
        ppo_config: DictConfig,
        device: torch.device
    ):
        self.policy = policy_net.to(device)
        self.device = device

        self.gamma: float = ppo_config['gamma']
        self.gae_labmda: float = ppo_config['gae_lambda']
        self.ppo_epsilon: float = ppo_config['ppo_epsilon']
        # self.critic_eps: float = ppo_config['critic_eps']

        self.actor_coef: float = ppo_config['actor_coef']
        self.critic_coef: float = ppo_config['critic_coef']
        self.entropy_coef: float = ppo_config['entropy_coef']

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.policy = nn.parallel.DistributedDataParallel(
            self.policy, device_ids=[device], output_device=device
        )

        self.init_seed: int = self.rank * 13_541

    def collect_rollouts(
        self,
        env: LuxObsWrapper,
        n_buffer_size: int,
        max_steps_env: int,
        shift_seed = 0,
    ) -> Tuple[RolloutBuffer, float, float]:
        self.policy.eval()
        rollout_buffer: RolloutBuffer = RolloutBuffer()

        step = 0
        total_rewards = []

        ore_gathered = []
        ice_gathered = []
        rubble_digged = []
        
        ore_transfered = []
        ice_transfered = []

        lichen_grow = []

        light_built = []
        light_destroyed = []
        heavy_built = []
        heavy_destroyed = []

        episod_len = []

        while len(rollout_buffer) < n_buffer_size:
            env_step = 0

            trajectory = []

            obs_dict, info = env.reset(seed=self.init_seed + shift_seed)
            self.init_seed += 1
            last_critic_value_if_not_done: Dict[str, npt.NDArray] = {
                agent_name: 0 for agent_name in obs_dict.keys()
            }
            total_rev = {agent: 0 for agent in obs_dict}

            while env_step <= max_steps_env:
                prev_obs_dict = obs_dict
                tensor_dict = dict_to_device(obs_dict, self.device)

                action_outs: Dict[str, Dict[str, Union[npt.NDArray, Tuple[npt.NDArray]]]] = dict()
                for agent_name, obs_tensor_dict in tensor_dict.items():
                    with torch.no_grad():
                        action_outs[agent_name] = tensor_out_to_numpy(
                            self.policy.forward(feats=obs_tensor_dict, action_sample=None)
                        )
                if env == max_steps_env:
                    # bootstrap value/advantage
                    for agent_name, dct in action_outs.values():
                        last_critic_value_if_not_done[agent_name] = dct["critic_value"]
                    break

                actions: Dict[str, Tuple[npt.NDArray]] = dict()
                for agent, dct in action_outs.items():
                    actions[agent] = dct["action_sample"]
                    # print(agent, dct["action_sample"][0].shape, len(dct["action_sample"]))
                obs_dict, reward_dict, done_dict, trunc_dict, info = env.step(actions)
                done_flag = False
                for done_value in done_dict.values():
                    if done_value:
                        done_flag = True
                        break

                # [S, A, R, S]
                trajectory += [(prev_obs_dict, action_outs, reward_dict, obs_dict, done_flag)]
                for agent in reward_dict:
                    total_rev[agent] += reward_dict[agent]

                env_step += 1
                if done_flag:
                    break

            game_stats = env.get_stats()
            for stats in game_stats.values():
                ore_gathered += [stats["generation"]["ore"]["LIGHT"] + stats["generation"]["ore"]["HEAVY"]]
                ice_gathered += [stats["generation"]["ice"]["LIGHT"] + stats["generation"]["ice"]["HEAVY"]]
                rubble_digged += [stats["destroyed"]["rubble"]["LIGHT"] + stats["destroyed"]["rubble"]["HEAVY"]]
                
                ore_transfered += [stats["transfer"]["ore"]]
                ice_transfered += [stats["transfer"]["ice"]]

                light_built += [stats["generation"]["built"]["LIGHT"]]
                heavy_built += [stats["generation"]["built"]["HEAVY"]]

                light_destroyed += [stats["destroyed"]["LIGHT"]]
                heavy_destroyed += [stats["destroyed"]["HEAVY"]]

            episod_len += [env_step]
            total_rewards.extend(list(total_rev.values()))
            # GAE
            next_critic_value = last_critic_value_if_not_done
            next_gae = {
                agent: 0 for agent in obs_dict.keys()
            }
            advantages = []
            returns = []
            critic_values = []

            for t in reversed(range(len(trajectory))):
                _, action_outs, reward_dict, _, done_flag = trajectory[t]
                current_critic_value = {agent: action_outs[agent]["critic_value"] for agent in action_outs}
                critic_values.append(current_critic_value)
                simple_advantage = {
                    agent: reward_dict[agent] + (
                        self.gamma * next_critic_value[agent]
                    ) - current_critic_value[agent] for agent in reward_dict
                }
                gae_advantage = {
                    agent: simple_advantage[agent] + (
                        self.gamma * self.gae_labmda * next_gae[agent]
                    ) for agent in reward_dict
                }
                advantages.append(gae_advantage)
                returns.append({
                    agent: current_critic_value[agent] + gae_advantage[agent]
                    for agent in reward_dict
                })

            advantages = reversed(advantages)
            returns = reversed(returns)
            critic_values = reversed(critic_values)
            rollout_buffer.add_trajectory(trajectory, critic_values, advantages, returns)

            step += 1

        rollout_buffer.shuffle_truncate(n_buffer_size, self.init_seed)

        return rollout_buffer, np.mean(total_rewards), np.mean(episod_len), \
                                np.mean(ore_gathered), np.mean(ice_gathered), \
                                np.mean(ore_transfered), np.mean(ice_transfered), \
                                np.mean(rubble_digged), \
                                np.mean(light_built), np.mean(heavy_built), \
                                np.mean(light_destroyed), np.mean(heavy_destroyed)

    def setup_optimizer(self):
        self.optim = torch.optim.AdamW(
            params=self.policy.parameters(),
            lr=1e-4
        )

    def ppo_actor_loss(
        self,
        unit_logprob: torch.Tensor, fact_logprob: torch.Tensor,
        unit_logprob_old: torch.Tensor, fact_logprob_old: torch.Tensor,
        unit_mask: torch.Tensor, fact_mask: torch.Tensor,
        gae_advantage: torch.Tensor
    ) -> torch.Tensor:
        batch_size = unit_mask.shape[0]

        unit_mask = unit_mask.sum(dim=1) > 0 # [B, H, W]
        fact_mask = fact_mask.sum(dim=1) > 0 # [B, H, W]

        unit_diff = unit_logprob - unit_logprob_old
        fact_diff = fact_logprob - fact_logprob_old

        unit_rho = torch.exp(unit_diff)
        fact_rho = torch.exp(fact_diff)

        unit_clipped_rho = torch.clip(unit_rho, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
        fact_clipped_rho = torch.clip(fact_rho, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)

        unit_min_ce = torch.minimum(unit_rho, unit_clipped_rho)
        fact_min_ce = torch.minimum(fact_rho, fact_clipped_rho)

        unit_min_ce = torch.sum(unit_min_ce * unit_mask, dim=(1, 2))
        fact_min_ce = torch.sum(fact_min_ce * fact_mask, dim=(1, 2))

        # unit_ce_rho = torch.sum(unit_rho * unit_mask, dim=(1, 2))# / torch.sum(unit_mask, dim=(1, 2))
        # fact_ce_rho = torch.sum(fact_rho * fact_mask, dim=(1, 2))

        # unit_ce_rho_clipped = torch.sum(unit_clipped_rho * unit_mask, dim=(1, 2))# / torch.sum(unit_mask, dim=(1, 2))
        # fact_ce_rho_clipped = torch.sum(fact_clipped_rho * fact_mask, dim=(1, 2))

        count_actions = torch.sum(unit_mask, dim=(1, 2)) + torch.sum(fact_mask, dim=(1, 2))

        # ce_rho = (unit_ce_rho + fact_ce_rho) / count_actions
        # ce_rho_clipped = (unit_ce_rho_clipped + fact_ce_rho_clipped) / count_actions

        ce = (unit_min_ce + fact_min_ce)# / count_actions
        gae_advantage = (gae_advantage - torch.mean(gae_advantage)) / (torch.std(gae_advantage) + 1e-8)
        gae_adv_ce = ce * gae_advantage

        # min_ce = torch.minimum(ce_rho, ce_rho_clipped)
        # gae_adv_ce = min_ce * gae_advantage

        return -torch.mean(gae_adv_ce)

    def ppo_critic_loss(
        self,
        critic_target: torch.Tensor,
        current_value: torch.Tensor,
        old_value: torch.Tensor
    ) -> torch.Tensor:
        simple = (critic_target - current_value) ** 2
        # clipped_diff = torch.clip(current_value - old_value, -self.critic_eps, self.critic_eps)
        # clipped_mse = (critic_target - old_value - clipped_diff) ** 2
        # loss = torch.maximum(simple, clipped_mse)
        loss = simple
        return torch.mean(loss)
    
    def ppo_entropy_loss(
        self,
        unit_mask: torch.Tensor, fact_mask: torch.Tensor,
        unit_entropy: torch.Tensor, fact_entropy: torch.Tensor
    ) -> torch.Tensor:
        unit_mask = unit_mask.sum(dim=1) > 0
        fact_mask = fact_mask.sum(dim=1) > 0
        num = (unit_entropy * unit_mask + fact_entropy * fact_mask).sum(dim=(1, 2))
        den = torch.sum(unit_mask, dim=(1, 2)) + torch.sum(fact_mask, dim=(1, 2))
        loss = num# / den
        return torch.mean(loss)

    def learn(
        self,
        n_ppo_iters: int,
        env: LuxObsWrapper,
        n_buffer_size: int,
        max_env_steps: int,
        ckpt_folder: str,
        n_train_epochs: int = 2,
        batch_size: int = 256,
        num_workers: int = 5,
        global_train_cfg = dict()
    ) -> None:
        self.setup_optimizer()
        step = 0

        if self.rank == 0:
            session = wandb.init(
                project="mars",
                name="long_simple_ppo_sum_tensor_lr=1e-4_phase0_maxF2_minF1", 
                config=global_train_cfg
            )

        bar_range = range if self.rank != 0 else trange

        for sim_idx in bar_range(n_ppo_iters):
            rollout_buffer, mean_total_rev, mean_episod_len, o_g, i_g, o_t, i_t, \
                rubble_digged, l_b, h_b, l_d, h_d = self.collect_rollouts(
                env, n_buffer_size, max_env_steps
            )

            mean_cat = torch.Tensor([
                mean_total_rev, mean_episod_len, o_g, i_g, o_t, i_t,
                rubble_digged, l_b, h_b, l_d, h_d
            ]).to(self.device)

            dist.reduce(mean_cat, dst=0, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                mean_cat /= self.world_size
                mean_total_rev, mean_episod_len, o_g, i_g, o_t, i_t, \
                    rubble_digged, l_b, h_b, l_d, h_d = mean_cat
                wandb.log({
                    'valid/mean_reward': mean_total_rev,
                    'valid/mean_episod_len': mean_episod_len,
                    'mean_digged/ore': o_g,
                    'mean_digged/ice': i_g,
                    'mean_digged/rubble': rubble_digged,
                    'mean_transfered/ore': o_t,
                    'mean_transfered/ice': i_t,
                    'mean_built/LIGHT': l_b,
                    'mean_built/HEAVY': h_b,
                    'mean_destroyed/LIGHT': l_d,
                    'mean_destroyed/HEAVY': h_d,
                }, step=step)

            dataset = TrajectoryDataset(rollout_buffer)
            rollout_buffer.reset_buffer()
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                drop_last=True
            )

            self.policy.train()
            for _ in range(n_train_epochs):
                for batch in dataloader:
                    batch = tensor_dict_to_device(batch, device=self.device)
                    feats = batch['observation']

                    actions_tuple = batch['action']
                    actions_out = self.policy.forward(feats, actions_tuple)

                    actor_loss = self.ppo_actor_loss(
                        *actions_out['action_logprob'],
                        *batch['action_logprob'],
                        feats['unit_mask'], feats['factory_mask'],
                        batch['gae_advantage']
                    )
                    critic_loss = self.ppo_critic_loss(
                        batch['critic_target'],
                        actions_out['critic_value'],
                        batch['critic_old']
                    )
                    entropy_loss = self.ppo_entropy_loss(
                        feats['unit_mask'], feats['factory_mask'],
                        *actions_out['action_entropy']
                    )
                    loss = self.actor_coef * actor_loss + self.critic_coef * critic_loss
                    loss += self.entropy_coef * entropy_loss

                    self.optim.zero_grad()
                    loss.backward()
                    w_norm, g_norm = get_w_grad_norm(self.policy)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optim.step()

                    log_cat = torch.cat(
                        [x[None] for x in [actor_loss, critic_loss, entropy_loss, loss, w_norm, g_norm]]
                    )
                    dist.reduce(log_cat, dst=0, op=dist.ReduceOp.SUM)

                    if self.rank == 0:
                        log_cat /= self.world_size
                        actor_loss, critic_loss, entropy_loss, loss, w_norm, g_norm = log_cat[:6]
                        wandb.log({
                            'train/actor_loss': actor_loss,
                            'train/critic_loss': critic_loss,
                            'train/entropy_loss': entropy_loss,
                            'train/loss': loss,
                            'train/weights_norm': w_norm,
                            'train/grad_norm': g_norm
                        })
                    step += 1

            if self.rank == 0:
                if not osp.exists(ckpt_folder):
                    import os
                    os.makedirs(ckpt_folder)
                path = osp.join(ckpt_folder, f"long_ckpt_{sim_idx}.pth")
                print(f'Saving ckpt at: {path}')
                torch.save(self.policy.state_dict(), path)
        if self.rank == 0:
            session.finish()
