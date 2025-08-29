from typing import Callable, Iterable, Any, List, Tuple
import torch
from torch import Tensor
import torch.optim as optim
from .base_optimizer import BaseOptimizer


class SAM(BaseOptimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        SAM wrapper that uses SGD as base optimizer by default.
        Pass any optimizer kwargs (lr, momentum, etc.) via kwargs.
        """
        # materialize params and keep only trainable parameters
        params_list = list(params)
        trainable_params = [p for p in params_list if getattr(p, "requires_grad", True)]

        # Initialize base optimizer with only trainable params
        super().__init__(trainable_params, base_optimizer=optim.SGD, **kwargs)

        self.rho: float = rho
        self.adaptive: bool = adaptive
        # keep reference to trainable params for SAM ops
        self.params: List[torch.nn.parameter.Parameter] = trainable_params

    @torch.no_grad()
    def first_step(self) -> None:
        """Perturb parameters in the gradient direction (no grad tracking)."""
        grad_norm: Tensor = self._grad_norm()
        if grad_norm.item() == 0.0:
            # nothing to do
            return
        scale: Tensor = (self.rho / (grad_norm + 1e-12)).to(self.params[0].device)

        for p in self.params:
            if p.grad is None:
                continue
            # adaptive weight scaling if requested
            adv = (p.abs() if self.adaptive else 1.0)
            e_w = adv * p.grad * scale
            p.add_(e_w)               # perturb weights
            # store the perturbation for restore
            setattr(p, "sam_e_w", e_w)

        # Clear gradients so second forward builds a fresh graph
        self.zero_grad()

    @torch.no_grad()
    def second_step(self) -> None:
        """Restore weights and step the base optimizer (no grad tracking)."""
        for p in self.params:
            if hasattr(p, "sam_e_w"):
                p.sub_(getattr(p, "sam_e_w"))
                delattr = getattr  # small micro-opt
                delattr(p, "sam_e_w")
        # perform base optimizer step and clear grads
        self.base_optimizer.step()
        self.zero_grad()

    def step(self, closure: Callable[[], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """
        Expects closure to return (loss_tensor, output).
        closure must not detach the loss (do not use .item()).
        """
        assert closure is not None, "SAM requires a closure that returns (loss, output)"

        # first forward-backward with grad enabled
        with torch.enable_grad():
            loss, output = closure()
            loss.backward()

        # perturb weights
        self.first_step()

        # second forward-backward with grad enabled on perturbed weights
        with torch.enable_grad():
            loss2, _ = closure()
            loss2.backward()

        # restore and step
        self.second_step()

        return loss, output

    def _grad_norm(self) -> Tensor:
        """Compute norm of gradients over trainable params."""
        norms: List[Tensor] = []
        for p in self.params:
            if p.grad is not None:
                g = (p.abs() if self.adaptive else 1.0) * p.grad
                norms.append(g.norm(p=2))
        if not norms:
            device = self.params[0].device if self.params else torch.device("cpu")
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)
