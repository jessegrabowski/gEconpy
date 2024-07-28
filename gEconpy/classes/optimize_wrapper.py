import logging
import sys

from collections.abc import Callable

import numpy as np

from fastprogress.fastprogress import ProgressBar, progress_bar
from scipy.optimize import OptimizeResult

_log = logging.getLogger(__name__)


class CostFuncWrapper:
    def __init__(
        self,
        f: Callable,
        f_jac: Callable | None = None,
        f_hess: Callable | None = None,
        maxeval: int = 5000,
        progressbar: bool = True,
        update_every: int = 10,
    ):
        self.n_eval = 0
        self.maxeval = maxeval
        self.f = f
        self.use_jac = False
        self.use_hess = False
        self.update_every = update_every
        self.interrupted = False
        self.desc = "f = {:,.5g}"

        if f_jac is not None:
            self.desc += ", ||grad|| = {:,.5g}"
            self.use_jac = True
            self.f_jac = f_jac

        if f_hess is not None:
            self.desc += ", ||hess|| = {:,.5g}"
            self.use_hess = True
            self.f_hess = f_hess

        self.previous_x = None
        self.progressbar = progressbar
        if progressbar:
            self.progress = progress_bar(
                range(maxeval), total=maxeval, display=progressbar
            )
            self.progress.update(0)
        else:
            self.progress = range(maxeval)

    def step(self, x, params):
        grad = None
        hess = None
        value = self.f(x, params)

        if self.use_jac:
            grad = self.f_jac(x, params)
            if self.use_hess:
                hess = self.f_hess(x, params)
            if np.all(np.isfinite(x)):
                self.previous_x = x
        else:
            self.previous_x = x

        if self.n_eval % self.update_every == 0:
            self.update_progress_desc(value, grad, hess)

        if self.n_eval > self.maxeval:
            self.update_progress_desc(value, grad, hess)
            self.interrupted = True

            if self.use_jac:
                return value, grad
            return value

        self.n_eval += 1
        if self.progressbar:
            assert isinstance(self.progress, ProgressBar)
            self.progress.update_bar(self.n_eval)

        if self.use_jac:
            if self.use_hess:
                return value, grad  # , hess
            else:
                return value, grad
        else:
            return value

    def __call__(self, x, params):
        try:
            return self.step(x, params)
        except (KeyboardInterrupt, StopIteration):
            self.interrupted = True
            return self.step(self.x, params)

    def callback(self, *args):
        if self.interrupted:
            raise StopIteration

    def update_progress_desc(
        self, value: float, grad: np.float64 = None, hess: np.float64 = None
    ) -> None:
        if isinstance(value, np.ndarray):
            value = (value**2).sum()
        elif isinstance(value, (list, tuple)):
            value = sum([x**2 for x in value])

        if self.progressbar:
            if grad is None:
                self.progress.comment = self.desc.format(value)
            else:
                if hess is None:
                    norm_grad = np.linalg.norm(grad)
                    self.progress.comment = self.desc.format(value, norm_grad)
                else:
                    norm_grad = np.linalg.norm(grad)
                    norm_hess = np.linalg.norm(hess)
                    self.progress.comment = self.desc.format(
                        value, norm_grad, norm_hess
                    )


def optimzer_early_stopping_wrapper(f_optim):
    objective = f_optim.args[0]
    progressbar = objective.progressbar

    res = f_optim()

    total_iters = objective.n_eval
    if progressbar:
        objective.progress.total = total_iters
        objective.progress.update(total_iters)
        print(file=sys.stdout)

    return res


def postprocess_optimizer_res(
    res,
    res_dict,
    f_resid,
    f_jac,
    verbose=True,
) -> tuple[dict, OptimizeResult]:
    success = res.success

    f_x = np.r_[[x.ravel() for x in f_resid(**res_dict)]]
    df_dx = f_jac(**res_dict)

    msg = (
        f"Steady state {'' if success else 'NOT '}found.\n"
        f'{"-"*80}\n'
        f"{'Optimizer message':<30}{res.message}\n"
        f"{'Sum of squared residuals':<30}{(f_x ** 2).sum()}\n"
        f"{'Maximum absoluate error':<30}{np.abs(f_x).max()}\n"
        f"{'Gradient norm at solution':<30}{np.linalg.norm(df_dx)}"
    )

    if verbose:
        _log.info(msg)

    return res_dict, success
