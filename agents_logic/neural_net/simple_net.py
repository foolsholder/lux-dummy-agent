import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from omegaconf import OmegaConf, DictConfig

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, padding=2):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(),
        )
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        # self.leaky_relu = nn.LeakyReLU()
        self.shortcut = nn.Sequential()
        self.selayer = SELayer(out_channel)
        self._init_w_b()

    def forward(self, x):
        out = self.left(x)
        out = self.selayer(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

    def _init_w_b(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _init_w_b(layers):
    for layer in layers:
        nn.init.kaiming_normal_(layer.weight)
        #nn.init.zeros_(layer.bias)

class SimpleCentralizedNet(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
    ):
        super().__init__()

        # print(model_param)
        # print(model_param['emb_dim'])

        emb_dim = int(model_cfg["emb_dim"])
        n_res_blocks = model_cfg["n_res_blocks"]

        res_channel = model_cfg["res_channel"]
        self.res_channel = res_channel

        self.map_channels = model_cfg.map_channel

        self.n_actions = model_cfg.n_actions
        self.embedding_layer = nn.Linear(model_cfg.global_feature_dims, emb_dim)

        input_channel = emb_dim + model_cfg.map_channel

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(res_channel, res_channel) for _ in range(n_res_blocks)
            ]
        )

        self.spectral_norm = nn.utils.spectral_norm(
            nn.Conv2d(res_channel,res_channel,kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.input_conv1 = nn.Conv2d(input_channel, res_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.robot_action = nn.Conv2d(res_channel, model_cfg.n_actions["robot"], kernel_size=1, stride=1, padding=0, bias=False)
        self.fact_action = nn.Conv2d(res_channel, model_cfg.n_actions["factory"], kernel_size=1, stride=1, padding=0, bias=False)

        _init_w_b([self.robot_action, self.fact_action])
        self.critic_fc = nn.Linear(res_channel, 1)
        nn.init.kaiming_normal_(self.input_conv1.weight)
        nn.init.xavier_normal_(self.critic_fc.weight)

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        action_sample: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor]]]:
        # print(feats.keys())

        map_feature = feats["map_info"]
        map_size = map_feature.shape[2]

        global_time = feats["global_time"]

        global_feature = self.embedding_layer(global_time)
        global_feature = global_feature.view(
            -1, global_feature.shape[1], 1, 1
        ).expand(-1, global_feature.shape[1], map_size, map_size)

        fff = torch.cat([global_feature, map_feature], dim=1)

        fff = self.input_conv1(fff)

        for block in self.res_blocks:
            fff = block(fff)

        fff = self.spectral_norm(fff)

        robot_action = self.robot_action(fff)
        fact_action = self.fact_action(fff)

        fff = torch.flatten(fff, start_dim=-2, end_dim=-1).mean(dim=-1)

        critic_value = self.critic_fc(fff.view(-1, self.res_channel)).view(-1)

        robot_mask = feats['unit_mask']
        fact_mask = feats['factory_mask']

        robot_action = torch.where(robot_mask, robot_action, 1e-8)
        fact_action = torch.where(fact_mask, fact_action, 1e-8)
        # critic_value = F.tanh(self.critic_fc(x.view(-1, self.res_channel))).view(-1)
        unit_dist = torch.distributions.Categorical(logits=robot_action.permute(0, 2, 3, 1))
        fact_dist = torch.distributions.Categorical(logits=fact_action.permute(0, 2, 3, 1))

        if action_sample is None:
            unit_sample = unit_dist.sample()
            fact_sample = fact_dist.sample()
        else:
            unit_sample, fact_sample = action_sample

        unit_logprob = unit_dist.log_prob(unit_sample)
        fact_logprob = fact_dist.log_prob(fact_sample)

        return {
            "action_logits": (robot_action, fact_action),
            "action_sample": (unit_sample, fact_sample),
            "action_logprob": (unit_logprob, fact_logprob),
            "action_entropy": (unit_dist.entropy(), fact_dist.entropy()),
            "critic_value": critic_value
        }
