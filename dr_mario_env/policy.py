"""Maskable CNN policy with a shared spatial trunk for Dr. Mario."""

from __future__ import annotations

from functools import partial
from math import prod
from typing import Any

import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class DrMarioCnnExtractor(BaseFeaturesExtractor):
    """Shared padded 3x3 stack that preserves an ``channels x 8 x 16`` map, then flattens."""

    def __init__(
        self,
        observation_space: spaces.Box,
        channels: int = 16,
        depth: int = 8,
    ) -> None:
        super().__init__(observation_space, features_dim=channels * prod(observation_space.shape[1:]))
        self.channels = channels
        in_channels = observation_space.shape[0]

        layers: list[nn.Module] = []
        for i in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(self.cnn(observations))


class _Actor1x1Head(nn.Module):
    """Map shared ``n x 8 x 16`` features to flat action logits via a 1x1 conv."""

    def __init__(self, channels: int, rotations: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, rotations, kernel_size=1)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        spatial = latent.view(-1, self.channels, 8, 16)
        return self.conv(spatial).reshape(spatial.shape[0], -1)


class _CriticHead(nn.Module):
    """Shrink spatially with unpadded 3x3s, max-pool to a vector, then a linear value."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, latent: th.Tensor) -> th.Tensor:
        spatial = latent.view(-1, self.channels, 8, 16)
        return self.net(spatial)


class DrMarioCnnPolicy(MaskableActorCriticPolicy):
    """Shared CNN trunk; actor is a 1x1 conv; critic pools the same spatial features."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", DrMarioCnnExtractor)
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))
        kwargs.setdefault("activation_fn", nn.ReLU)
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        channels = self.features_extractor.channels
        self.action_net = _Actor1x1Head(channels)
        self.value_net = _CriticHead(channels)
        if self.ortho_init:
            self.action_net.apply(partial(self.init_weights, gain=0.01))
            self.value_net.apply(partial(self.init_weights, gain=1.0))
        # Re-create optimizer so the replaced heads are included.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
