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
        self.A = nn.Parameter(torch.empty(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        self.reset_LoRA_parameters()

    def reset_LoRA_parameters(self):
        # mode = 'fan_in' directs the initialization formula to use the number of input features for the given layer
        init.kaiming_uniform_(self.A, a = 0, mode = 'fan_in', nonlinearity = 'relu')

    def forward(self, x):
        base_output = self.linear_layer(x)
        Ax = F.linear(x, self.A)
        BAx = F.linear(Ax, self.B)
        adapter_output = BAx * self.scaling
        lora_out = base_output + adapter_output

        return lora_out

class DummyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.feed_forward = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DummyTransformerBlock()
        self.layer2 = DummyTransformerBlock()
        self.lm_head = nn.Linear(512, 1024)


def find_linear_layers(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):

            if '.' not in name:
                parent_module = model
                child = name
            else:
                path_parts = name.rsplit('.', 1)
                child = path_parts[-1]
                parent = path_parts[0]
                steps = parent.split('.')
                parent_module = model

                for step in steps:
                    parent_module = getattr(parent_module, step)
                    
            new_LoRA_layer = LoRALinear(4, 4, module)
            setattr(parent_module, child, new_LoRA_layer)

            print(f'Target: {name} | Parent Type: {type(parent_module)} | Child Name: {child}')
            print('--------------------------------------------------')
                    

dummy_model = DummyModel()
find_linear_layers(dummy_model)