"""A stand-in for recurrent.LSTMGaussianPolicy with the attributes actor_step reads (recurrent.py imports skrl and
sits in the Isaac task package, which these CPU tests do not import)."""

import torch
import torch.nn as nn


class StandInLSTMPolicy(nn.Module):
    def __init__(self, num_obs: int, num_actions: int, hidden_size: int = 16, num_layers: int = 1,
                 layers=(32,), skip_observations: bool = False, cell_cap: float = 0.0):
        super().__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.skip_observations, self.cell_cap = skip_observations, float(cell_cap)
        self.lstm = nn.LSTM(num_obs, hidden_size, num_layers, batch_first=True)
        mlp, width = [], hidden_size + (num_obs if skip_observations else 0)
        for size in layers:
            mlp += [nn.Linear(width, size), nn.ELU()]
            width = size
        self.net = nn.Sequential(*mlp, nn.Linear(width, num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(num_actions))

    def compute(self, inputs, role=""):
        """recurrent.LSTMGaussianPolicy.compute's acting branch (no "terminated" in inputs), copied line for line."""
        observations = inputs["observations"]
        h, c = inputs["rnn"]
        out, (h, c) = self.lstm(observations.unsqueeze(1), (h, c))
        if self.cell_cap > 0.0:
            c = self.cell_cap * torch.tanh(c / self.cell_cap)
        features = out.squeeze(1)
        if self.skip_observations:
            features = torch.cat([features, observations], dim=-1)
        return self.net(features), {"log_std": self.log_std_parameter, "rnn": [h, c]}
