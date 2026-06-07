import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LoRALinear(nn.Module):
    def __init__(self, alpha, r, linear_layer):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.linear_layer = linear_layer
        self.linear_layer.weight.requires_grad = False
        if self.linear_layer.bias is not None:
            self.linear_layer.bias.requires_grad = False
        self.in_features = self.linear_layer.in_features
        self.out_features = self.linear_layer.out_features
        self.A = nn.Parameter(torch.empty(r, self.in_featrues))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        self.reset_LoRA_parameters()

    def reset_LoRA_parameters(self):
        init.kaiming_uniform_(self.A, a = 0, mode = 'fan_in', nonlinearity = 'relu')

    def forward(self, x):
        base_output = self.linear_layer(x)
        Ax = F.linear(x, self.A)
        BAx = F.linear(Ax, self.B)
        adapter_output = BAx * self.scaling
        lora_out = base_output + adapter_output

        return lora_out