import torch

from third_party.RAdam.radam import RAdam
from datamodule.encoding import grf_coding, stochastic_position_coding, position_coding
from datamodule.normalization import minmax, standardize, uniformize
from modelmodule.norse_model import SNN as NorseSNN
from modelmodule.model import RNN
# from third_party.lookahead_pytorch.optimizer import Lookahead

# TODO: Maybe add option for Lookahead?

optimizers = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RAdam": RAdam
}

loss_functions = {
    "MSELoss": torch.nn.MSELoss, 
    "SmoothL1Loss": torch.nn.SmoothL1Loss, 
    "L1Loss": torch.nn.L1Loss
}

encodings = {
    "grf": grf_coding,
    "stochastic": stochastic_position_coding,
    "position": position_coding
}

normalizations = {
    "minmax": minmax, 
    "standardize": standardize, 
    "uniformize": uniformize
}

network_types = {
    "norse_recurrent_snn": NorseSNN,
    "gru": RNN
}
