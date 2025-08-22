from .categorical_crossentropy import loss_fn as categorical_crossentropy_fn
from typing import Callable


loss_fns: dict[str, Callable] = {
    "categorical_crossentropy": categorical_crossentropy_fn
}