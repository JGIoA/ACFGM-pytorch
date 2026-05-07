from typing import List, Optional, Sequence
import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _get_scalar_dtype,
)

__all__ = ["ACFGM"]


def _tensor_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value.reshape(-1), ord=2)


class ACFGM(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        beta: float = 0.1,
        eps: float = 1e-8,
        lims: Optional[Sequence[float]] = None,
        linesearch: bool = False,
    ):
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # Corollary 2 beta \in (0, 1 - 3^0.5 / 2]
        # if not 0.0 <= beta < 0.134:
        if not 0.0 < beta < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {beta}")
        if lims is None:
            lims = (-1.0, 1.0)
        if len(lims) != 2 or lims[0] > lims[1]:
            raise ValueError(f"Invalid projection limits: {lims}")

        defaults = dict(
            beta=beta,
            eps=eps,
            lims=tuple(lims),
        )
        super().__init__(params, defaults)
        self.linesearch = linesearch
        if linesearch:
            self.acfgm_fn = acfgm_Lsearch
        else:
            self.acfgm_fn = acfgm_noLsearch


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    p_state["step"] = torch.tensor(
                        float(p_state["step"]), dtype=_get_scalar_dtype()
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        old_grads,
        old_old_grads,
        old_old_params,
        old_old_objs,
        old_ys,
        old_taus,
        old_old_taus,
        old_etas,
        L0s,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                old_grads.append(p.grad)
                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    state["old_old_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=p.device
                    )
                    state["old_old_param"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=p.device
                    )
                    state["old_old_obj"] = torch.zeros(
                        (), dtype=_get_scalar_dtype(), device=p.device
                    )

                    state["old_y"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ).copy_(p)
                    state["old_tau"] = (
                        torch.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        * -1.0
                    )
                    state["old_old_tau"] = (
                        torch.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        * -1.0
                    )
                    state["old_eta"] = (
                        torch.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        * -1.0
                    )
                    state["L0"] = torch.tensor(
                        1.0, dtype=_get_scalar_dtype(), device=p.device
                    )

                state["step"] += 1
                old_old_grads.append(state["old_old_grad"])
                old_old_params.append(state["old_old_param"])
                old_old_objs.append(state["old_old_obj"])
                old_ys.append(state["old_y"])
                old_taus.append(state["old_tau"])
                old_old_taus.append(state["old_old_tau"])
                old_etas.append(state["old_eta"])
                L0s.append(state["L0"])

        return has_complex

    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        closure = torch.enable_grad()(closure)
        old_obj = closure()
        old_obj_for_update = old_obj.detach().sum()

        for group in self.param_groups:
            params_with_grad = []
            old_grads = []
            old_old_grads = []
            old_old_params = []
            old_old_objs = []
            old_ys = []
            old_taus = []
            old_old_taus = []
            old_etas = []
            L0s = []
            beta = group["beta"]
            eps = group["eps"]
            lims = group["lims"]

            self._init_group(
                group,
                params_with_grad,
                old_grads,
                old_old_grads,
                old_old_params,
                old_old_objs,
                old_ys,
                old_taus,
                old_old_taus,
                old_etas,
                L0s,
            )

            self.acfgm_fn(
                params_with_grad,
                old_grads,
                old_old_grads,
                old_old_params,
                old_old_objs,
                old_ys,
                old_taus,
                old_old_taus,
                old_etas,
                beta,
                eps,
                old_obj_for_update,
                lims,
                L0s,
            )

        return old_obj


def update(
    eta: Tensor,
    tau: Tensor,
    old_y: Tensor,
    old_grad: Tensor,
    old_param: Tensor,
    beta: float,
    lims: Sequence[float],
):
    # (eq 2.2) z_t = proj_X{y_t-1 - eta_t * g(x_t-1)}
    z = torch.sub(old_y, old_grad.mul(eta)).clamp(lims[0], lims[1])
    # (eq 2.3) y_t = (1 - beta_t) * y_t-1 + beta_t * z_t, then store y_t
    old_y.lerp_(end=z, weight=beta)
    # (eq 2.4) x_t = (z_t + tau_t * x_t-1) / (1 + tau_t)
    # x_t = (1 - 1/(1 + tau_t) * x_t-1) + (1/(1 + tau_t) * z_t)
    # with gamma = 1/(1 + tau_t)
    gamma = 1 / (1 + tau)
    param = torch.lerp(old_param, z, gamma)
    return param


def acfgm_Lsearch(
    params: List[Tensor],
    old_grads: List[Tensor],
    old_old_grads: List[Tensor],
    old_old_params: List[Tensor],
    old_old_objs: List[Tensor],
    old_ys: List[Tensor],
    old_taus: List[Tensor],
    old_old_taus: List[Tensor],
    old_etas: List[Tensor],
    beta: float,
    eps: float,
    old_obj: Tensor,
    lims: Sequence[float],
    L0s: List[Tensor],
):
    # step t
    for i, param in enumerate(params):
        # get x and g(x)
        old_param = param.detach().clone()  # x_t-1
        old_old_param = old_old_params[i]  # x_t-2
        old_grad = old_grads[i]  # g(x_t-1)
        old_old_grad = old_old_grads[i]  # g(x_t-2)

        # get optimizer states
        old_y = old_ys[i]  # y_t-1
        old_old_obj = old_old_objs[i]  # f_t-2
        old_tau = old_taus[i]  # tau_t-1
        old_old_tau = old_old_taus[i]  # tau_t-2
        old_eta = old_etas[i]  # eta_t-1
        device = param.device
        old_obj_param = old_obj.to(device=device, dtype=old_old_obj.dtype)

        # t = 1, 2
        if (old_old_tau == -1.0).any():
            L0 = L0s[i]
            # t = 1, f_t-1 = -1
            if (old_tau == -1.0).any():
                # Corollary 2: eta_1 \in [ beta / (4 * (1-beta) * L_1), 1 / (3 * L_1) ]
                # using upper lim for now
                eta = (
                    torch.ones_like(old_old_obj, device=device) * 1 / (3 * L0)
                )  # eta_1, init guess L = 1
                tau = torch.zeros_like(old_old_obj, device=device)  # tau_1

                # update optimizer state
                old_old_param.copy_(old_param)
                old_old_grad.copy_(old_grad)
                old_old_obj.copy_(old_obj_param)
                old_old_tau.copy_(old_tau)
                old_tau.copy_(tau)
                old_eta.copy_(eta)

                # acfgm udpate for x_t-1 -> x_t
                param.copy_(update(eta, tau, old_y, old_grad, old_param, beta, lims))

            # Corollary 2: t = 2, eta_2 = beta / (2 * L_1)
            else:
                # (2.5) L_1 = \|g_1 - g_0 \| / \|x_1 - x_0 \|
                param_delta_norm = _tensor_norm(old_param - old_old_param).clamp_min(
                    eps
                )
                old_L = (
                    _tensor_norm(old_grad - old_old_grad)
                    .div(param_delta_norm)
                    .clamp_min(eps)
                )  # L_1

                # Corollary 2: eta_1 \in [ beta / (4 * (1-beta) * L_1), 1 / (3 * L_1) ]
                eta_ulim = 1 / (3 * old_L)
                eta_llim = beta / (4 * (1 - beta) * old_L)
                if ((eta_llim <= old_eta) & (old_eta <= eta_ulim)).all():
                    eta = 1 / (2 * old_L)  # eta_2
                    tau = torch.ones_like(old_old_obj, device=device) * 2  # tau_2

                    # update optimizer state
                    old_old_param.copy_(old_param)
                    old_old_grad.copy_(old_grad)
                    old_old_obj.copy_(old_obj_param)
                    old_old_tau.copy_(old_tau)
                    old_tau.copy_(tau)
                    old_eta.copy_(eta)

                    # acfgm udpate for x_t-1 -> x_t
                    param.copy_(
                        update(eta, tau, old_y, old_grad, old_param, beta, lims)
                    )

                else:
                    if old_eta.max() <= eta_llim:
                        L0.div_(2)
                    else:
                        L0.mul_(2)

                    # revert to step 0
                    old_tau.copy_(torch.ones_like(old_old_obj, device=device) * -1.0)
                    old_old_tau.copy_(
                        torch.ones_like(old_old_obj, device=device) * -1.0
                    )
                    param.copy_(old_old_param)
                    old_y.copy_(old_old_param)

        # t >= 3
        else:
            # compute L_t-1
            # (eq 2.6) f_t-2 - f_t-1 - g_t-1 @ (x_t-2 - x_t-1)
            denom = (
                old_old_obj
                - old_obj_param
                - torch.sum(old_grad.mul(old_param - old_old_param))
            )
            # (eq 2.6) L=t-1 = \| g_t-1 - g_t-2 \|^2 / (2 * denom) if denom > 0 , else 0
            old_L = torch.where(
                denom > eps,
                _tensor_norm(old_grad - old_old_grad)
                .pow(2)
                .div(2 * denom.clamp_min(eps)),
                torch.zeros_like(old_old_obj),
            )

            # (eq 2.30) eta_t = min{ (tau_t-2 + 1) / tau_t-1 * eta_t-1), beta * tau_t-1 / (4 * L_t-1) }
            safe_old_L = old_L.clamp_min(eps)
            eta = torch.minimum(
                (old_old_tau + 1) / old_tau * old_eta,
                beta * old_tau * 0.25 / safe_old_L,
            )
            # (eq 2.31) tau_t = tau_t-1 + 2 * eta_t * L_t-1 / (beta * tau_t-1), with alpha = 0
            tau = old_tau + 2 * eta * old_L / (beta * old_tau)

            # update optimizer state
            old_old_param.copy_(old_param)
            old_old_grad.copy_(old_grad)
            old_old_obj.copy_(old_obj_param)
            old_old_tau.copy_(old_tau)
            old_tau.copy_(tau)
            old_eta.copy_(eta)

            # acfgm udpate for x_t-1 -> x_t
            param.copy_(update(eta, tau, old_y, old_grad, old_param, beta, lims))



def acfgm_noLsearch(
    params: List[Tensor],
    old_grads: List[Tensor],
    old_old_grads: List[Tensor],
    old_old_params: List[Tensor],
    old_old_objs: List[Tensor],
    old_ys: List[Tensor],
    old_taus: List[Tensor],
    old_old_taus: List[Tensor],
    old_etas: List[Tensor],
    beta: float,
    eps: float,
    old_obj: Tensor,
    lims: Sequence[float],
    L0s: List[Tensor],
):
    # step t
    for i, param in enumerate(params):
        # get x and g(x)
        old_param = param.detach().clone()  # x_t-1
        old_old_param = old_old_params[i]  # x_t-2
        old_grad = old_grads[i]  # g(x_t-1)
        old_old_grad = old_old_grads[i]  # g(x_t-2)

        # get optimizer states
        old_y = old_ys[i]  # y_t-1
        old_old_obj = old_old_objs[i]  # f_t-2
        old_tau = old_taus[i]  # tau_t-1
        old_old_tau = old_old_taus[i]  # tau_t-2
        old_eta = old_etas[i]  # eta_t-1
        device = param.device
        old_obj_param = old_obj.to(device=device, dtype=old_old_obj.dtype)

        # t = 1, 2
        if (old_old_tau == -1.0).any():
            # t = 1, f_t-1 = -1
            if (old_tau == -1.0).any():
                # eta_1 \in [ beta / (4 * (1-beta) * L_1), 1 / (3 * L_1) ]
                # using upper lim for now
                eta = torch.ones_like(old_old_obj, device=device) * 1 / (3 * 2)  # eta_1
                tau = torch.zeros_like(old_old_obj, device=device)  # tau_1

            # t = 2, eta_2 = beta / (2 * L_1)
            else:
                # (2.5) L_1 = \|g_1 - g_0 \| / \|x_1 - x_0 \|
                param_delta_norm = _tensor_norm(old_param - old_old_param).clamp_min(
                    eps
                )
                old_L = (
                    _tensor_norm(old_grad - old_old_grad)
                    .div(param_delta_norm)
                    .clamp_min(eps)
                )  # L_1
                eta = (1 / (2 * old_L)).clamp(0, 1)  # eta_2 tmp bounded for small old_L
                tau = torch.ones_like(old_old_obj, device=device) * 2  # tau_2

        # t >= 3
        else:
            # compute L_t-1
            # (eq 2.6) f_t-2 - f_t-1 - g_t-1 @ (x_t-2 - x_t-1)
            denom = (
                old_old_obj
                - old_obj_param
                - torch.sum(old_grad.mul(old_param - old_old_param))
            )
            # (eq 2.6) L=t-1 = \| g_t-1 - g_t-2 \|^2 / (2 * denom) if denom > 0 , else 0
            old_L = torch.where(
                denom > eps,
                _tensor_norm(old_grad - old_old_grad)
                .pow(2)
                .div(2 * denom.clamp_min(eps)),
                torch.zeros_like(old_old_obj, device=device),
            )

            # (eq 2.30) eta_t = min{ (tau_t-2 + 1) / tau_t-1 * eta_t-1), beta * tau_t-1 / (4 * L_t-1) }
            safe_old_L = old_L.clamp_min(eps)
            eta = torch.minimum(
                (old_old_tau + 1) / old_tau * old_eta,
                beta * old_tau * 0.25 / safe_old_L,
            )
            # (eq 2.31) tau_t = tau_t-1 + 2 * eta_t * L_t-1 / (beta * tau_t-1), with alpha = 0
            tau = old_tau + 2 * eta * old_L / (beta * old_tau)

        # update optimizer state
        old_old_param.copy_(old_param)
        old_old_grad.copy_(old_grad)
        old_old_obj.copy_(old_obj_param)
        old_old_tau.copy_(old_tau)
        old_tau.copy_(tau)
        old_eta.copy_(eta)

        # acfgm udpate for x_t-1 -> x_t
        param.copy_(update(eta, tau, old_y, old_grad, old_param, beta, lims))
