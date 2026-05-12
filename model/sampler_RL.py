# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch.nn import functional as F
from tqdm import tqdm

from flow_matching.path import MixtureDiscreteProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper
from flow_matching.solver.utils import get_nearest_times


def compute_token_transition_probability(
    x_t: torch.Tensor,              
    p_1t: torch.Tensor,              
    k_t: torch.Tensor,              
    d_k_t: torch.Tensor,            
    h: torch.Tensor,               
    vocabulary_size: int            
) -> torch.Tensor:
    if torch.isnan(p_1t).any() or torch.isinf(p_1t).any():
        p_1t = torch.nan_to_num(p_1t, nan=0.0, posinf=0.0, neginf=0.0)
        p_1t_sum = p_1t.sum(dim=-1, keepdim=True)
        p_1t = torch.where(p_1t_sum > 1e-8, p_1t / p_1t_sum, 1.0 / vocabulary_size)
    
    batch_size, seq_len = x_t.shape
    
    if k_t.dim() == 0:
        k_t = k_t.expand(batch_size, seq_len)
    if d_k_t.dim() == 0:
        d_k_t = d_k_t.expand(batch_size, seq_len)
    if h.dim() == 0:
        h = h.expand(batch_size, seq_len)
    
    lambda_i = d_k_t / (1 - k_t + 1e-8)  # [batch_size, seq_len]
    prob_matrix = torch.zeros(batch_size, seq_len, vocabulary_size, device=x_t.device)
    
    for i in range(vocabulary_size):
        x_1 = torch.full((batch_size,seq_len), i, dtype=torch.long, device=x_t.device)  # [batch_size]
        mask_change_possible = (x_1 != x_t).float()  # [batch_size, seq_len]
        # print(x_1, x_t, mask_change_possible)
        
        prob_stay = torch.exp(-h * lambda_i)  # [batch_size, seq_len]
        prob_stay = prob_stay * mask_change_possible + (1 - mask_change_possible)
        
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len).view(1, -1).expand(batch_size, -1)
        matrix = torch.zeros(batch_size, seq_len, vocabulary_size, device=x_t.device)
        matrix[batch_indices, seq_indices, x_1] = (1 - prob_stay) * mask_change_possible
        # print(matrix)
        weight = p_1t[:,:, i] # [batch_size]
        # print(i,prob_stay[0,0],k_t,d_k_t)
        
        matrix[batch_indices, seq_indices, x_t] = prob_stay
        # print(matrix)
        # matrix[batch_indices, seq_indices, x_1] = matrix[batch_indices, seq_indices, x_1] * weight
        # matrix[batch_indices, seq_indices, x_t] = matrix[batch_indices, seq_indices, x_t] * weight
        matrix = matrix * weight.unsqueeze(-1)  # [batch_size, seq_len, vocabulary_size]
        # print(matrix)
        prob_matrix += matrix
    # print('m',prob_matrix)
    min_prob = 1e-10
    prob_matrix = torch.clamp(prob_matrix, min=min_prob)
    
    prob_matrix = prob_matrix / (prob_matrix.sum(dim=-1, keepdim=True) + 1e-8)
        
    return prob_matrix


def apply_temperature_to_probs(probs: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:

    probs = torch.clamp(probs, min=1e-10) 
    safe_temperature = torch.clamp(temperature, min=1e-8)
    
    logits = torch.log(probs)
    
    if safe_temperature.dim() < logits.dim():
        safe_temperature = safe_temperature.view(-1, *([1] * (logits.dim() - 1)))
        
    scaled_logits = logits / safe_temperature
    
    new_probs = torch.softmax(scaled_logits, dim=-1)

    return new_probs



def get_flow_grpo_inspired_schedule(
    t: torch.Tensor, 
    alpha: float = 0.7 
) -> torch.Tensor:

    assert alpha >= 0.0, "alpha must be non-negative."
    
    t_clamped = torch.clamp(t, 0.0, 1.0)
    
    return 1.0 + alpha * torch.sqrt(1.0 - t_clamped)




class MixtureDiscreteEulerSolver(Solver):
    r"""Solver that simulates the CTMC process :math:`(X_t)_{t_{\text{init}}\leq t\leq t_{\text{final}}}` defined by :math:`p_t` the marginal probability path of ``path``.
    Given :math:`X_t \sim p_t`, the algorithm of solver step from :math:`t` to :math:`t+h` for the i-th coordinate is:

    .. math::

        \begin{align*}
            & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
            & \lambda^i \gets \sum_{x^i\ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
            & Z^i_{\text{change}} \sim U[0,1]\\
            & X_{t+h}^i \sim \begin{cases}
                \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) \text{ if $Z^i_{\text{change}}\le 1-e^{-h\lambda^i}$}\\
                \delta_{X_t^i}(\cdot) \text{ else }
            \end{cases}
        \end{align*}

    Where :math:`p_{1|t}(\cdot|X_t)` is the output of ``model``, and the conditional probability velocity is of the mixture probability path is:

    .. math::

        u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],

    where

    .. math::
        \hat{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{1-\kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right],

    and

    .. math::

        \check{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{\kappa_t}\left[ \delta_{y^i}(x^i) - p(x^i) \right].

    The source distribution :math:`p(x^i)` is given by ``p``.

    Args:
        model (ModelWrapper): trained with x-prediction, outputting posterior probabilities (in the range :math:`[0,1]`), output must be [..., vocabulary_size].
        path (MixtureDiscreteProbPath): Probability path used for x-prediction training.
        vocabulary_size (int): size of the discrete vocabulary.
        source_distribution_p (Optional[Tensor], optional): Source distribution, must be of shape [vocabulary_size]. Required only when divergence-free term for the probability velocity is non-zero. Defaults to None.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."

        self.source_distribution_p = source_distribution_p

    # @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        train_batchsize: int,
        num_samples: int,
        # temperature: float = 1.1,
        alpha: float = 0.7,
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Sample a sequence of discrete values from the given model.

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import MixtureDiscreteEulerSolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return ...

            model = DummyModel()
            solver = MixtureDiscreteEulerSolver(model=model)

            x_init = torch.LongTensor([122, 725])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): The initial state.
            step_size (Optional[float]): If float then time discretization is uniform with the given step size. If None then time discretization is set to be time_grid.
            div_free (Union[float, Callable[[float], float]]): The coefficient of the divergence-free term in the probability velocity. Can be either a float or a time dependent function. Defaults to 0.0.
            dtype_categorical (torch.dtype): Precision to use for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): The CTMC process is solved in the interval [time_grid[0], time_grid[-1]] and if step_size is None then time discretization is set by the time grid. Defaults to torch.tensor([0.0,1.0]).
            return_intermediates (bool): If True then return intermediate time steps according to time_grid. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

            if return_intermediates:
                # get order of intermediate steps:
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        steps_counter = 0
        res = []

        log_probs = []
        ref_log_probs = []
        kls = []
        all_x_t = []
        x_record = x_init.clone()
        # x_record = x_record.view((num_samples,train_batchsize, -1)).permute(1, 0, 2)
        all_x_t.append(x_record)
        t_record = []
        p_1t_record = []
        p_1t_ref_record = []


        if return_intermediates:
            res = [x_init.clone()]

        if verbose:
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                #
                t_record.append(t)

                # Sample x_1 ~ p_1|t( \cdot |x_t)
                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                #
                with torch.no_grad():
                    p_1t_ref = self.ref_model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)

                seq_len = p_1t.shape[1]
                vocabulary_size = p_1t.shape[2]
                p_1t_record.append(p_1t.view((train_batchsize, num_samples, seq_len, vocabulary_size)))
                p_1t_ref_record.append(p_1t_ref.view((train_batchsize, num_samples, seq_len, vocabulary_size)))

                current_temperature = get_flow_grpo_inspired_schedule(t, alpha=alpha)
                p_1t = apply_temperature_to_probs(p_1t, current_temperature)


                try:
                    x_1 = categorical(p_1t.to(dtype=dtype_categorical))
                except RuntimeError as e:
                    print(e)


                # Checks if final step
                if i == n_steps - 1:
                    # x_t = x_1
                    x_t = x_1.clone()

                    log_prob = torch.gather(torch.log(p_1t), dim=2, index=x_t.unsqueeze(-1)).squeeze(-1)
                    log_prob = log_prob.view((num_samples, train_batchsize, -1)).permute(1, 0, 2)
                    ref_log_prob = torch.gather(torch.log(p_1t_ref), dim=2, index=x_t.unsqueeze(-1)).squeeze(-1)
                    ref_log_prob = ref_log_prob.view((num_samples,train_batchsize, -1)).permute(1, 0, 2)
                else:
                    # Compute u_t(x|x_t,x_1)
                    scheduler_output = self.path.scheduler(t=t)

                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    # delta_1 = F.one_hot(x_1, num_classes=self.vocabulary_size).to(
                    #     k_t.dtype
                    # )
                    # u = d_k_t / (1 - k_t) * delta_1

                    # 
                    probmatrix = compute_token_transition_probability(
                        x_t=x_t,
                        p_1t=p_1t,
                        k_t=k_t,
                        d_k_t=d_k_t,
                        h=h,
                        vocabulary_size=self.vocabulary_size
                    )
                    ref_probmatrix = compute_token_transition_probability(
                        x_t=x_t,
                        p_1t=p_1t_ref,
                        k_t=k_t,
                        d_k_t=d_k_t,
                        h=h,
                        vocabulary_size=self.vocabulary_size
                    )


                    x_t = categorical(probmatrix.to(dtype=dtype_categorical))


                    log_prob = torch.gather(torch.log(probmatrix), dim=2, index=x_t.unsqueeze(-1)).squeeze(-1)
                    log_prob = log_prob.view((num_samples,train_batchsize, -1)).permute(1, 0, 2)
                    ref_log_prob = torch.gather(torch.log(ref_probmatrix), dim=2, index=x_t.unsqueeze(-1)).squeeze(-1)
                    ref_log_prob = ref_log_prob.view((num_samples,train_batchsize, -1)).permute(1, 0, 2)

                steps_counter += 1
                t = t + h

                log_probs.append(log_prob)
                ref_log_probs.append(ref_log_prob)
                x_record = x_t.clone()
                all_x_t.append(x_record)

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        all_x_t = torch.stack(all_x_t, dim=0)
        log_probs = torch.stack(log_probs, dim=0)
        ref_log_probs = torch.stack(ref_log_probs, dim=0)

        p_1t_record = torch.stack(p_1t_record, dim=0)
        p_1t_ref_record = torch.stack(p_1t_ref_record, dim=0)


        if return_intermediates:
            if step_size is None:
                return torch.stack(res, dim=0)
            else:
                return torch.stack(res, dim=0)[order]
        else:
            return x_t, log_probs, ref_log_probs, all_x_t, t_record, p_1t_record, p_1t_ref_record
