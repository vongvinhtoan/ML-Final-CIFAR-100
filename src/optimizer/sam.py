from typing import Callable, Type, Iterable, Any
import torch
from torch import Tensor
from torch.optim import Optimizer
from .base_optimizer import BaseOptimizer


class SAM(BaseOptimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        base_optimizer_cls: Type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, base_optimizer=base_optimizer_cls, **kwargs)
        self.rho: float = rho
        self.adaptive: bool = adaptive
        self.params: list[torch.nn.parameter.Parameter] = list(params)

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm: Tensor = self._grad_norm()
        scale: Tensor = self.rho / (grad_norm + 1e-12)

        for p in self.params:
            if p.grad is None:
                continue
            e_w: Tensor = (
                (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
            )
            p.add_(e_w)               # perturb weights
            p.sam_e_w = e_w           # type: ignore

    @torch.no_grad()
    def second_step(self) -> None:
        for p in self.params:
            if hasattr(p, "sam_e_w"):
                p.sub_(p.sam_e_w)     # type: ignore
                del p.sam_e_w         # type: ignore
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure: Callable[[], tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        assert closure is not None, "SAM requires a closure for re-evaluating loss"

        # first forward-backward
        loss, output = closure()
        loss.backward()
        self.first_step()

        # second forward-backward
        loss2, _ = closure()
        loss2.backward()
        self.second_step()

        return loss, output

    def _grad_norm(self) -> Tensor:
        norms: list[Tensor] = []
        for p in self.params:
            if p.grad is not None:
                g: Tensor = (torch.abs(p) if self.adaptive else 1.0) * p.grad
                norms.append(g.norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=self.params[0].device)
        return torch.norm(torch.stack(norms), p=2)
