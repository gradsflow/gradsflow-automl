from typing import Callable, Dict

from torch import nn

from gradsflow.utility.common import module_to_cls_index

nn_classes = module_to_cls_index(nn)


losses: Dict[str, Callable] = {k: v for k, v in nn_classes.items() if "loss" in k}
