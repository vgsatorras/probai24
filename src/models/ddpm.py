from typing import Optional, Tuple, TypeVar

import numpy as np
import torch

T = TypeVar("T", bound=torch.Tensor)


def broadcast_like(z: T, like: Optional[torch.Tensor]) -> T:
    """
    Add broadcast dimensions to x so that it can be broadcast over ``like``
    """
    if like is None:
        return z
    return z[(...,) + (None,) * (like.ndim - z.ndim)]


class DDPM(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        noise_schedule_type: str = "linear",
        N: int = 1000,
        t_epsilon: float = 0.001,
    ):
        super().__init__()
        self.N = N
        self.t_epsilon = t_epsilon
        self.model = model

        self._betas, self._alphas, self._alpha_bars = (
            torch.nn.Parameter(x, requires_grad=False)
            for x in self.get_coefs(N, noise_schedule_type)
        )

    # Setting up the betas, alphas, and alpha_bars coefficients

    @staticmethod
    def get_coefs(
        N: int, type: str = "linear"
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Get the coefficients for the noise schedule.

        Args:
            N (int): number of steps in the diffusion process.
            type (str, optional): type of noise schedule. Defaults to "linear".

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: 
                betas, alphas, alpha_bars
        """        
        if type == "linear":
            # TO DO: 
            # Set the betas to be linearly spaced between beta_min=0.0001 and beta_max=0.02
            # Set the alphas and alpha_bars accordingly
            raise NotImplementedError
        elif type == "cosine":
            # setting alpha_bars using the cosine schedule, and then computing alphas and betas
            # accordingly
            def cos2(t, s=0.001):
                return torch.cos((t / N + s) / (1 + s) * np.pi / 2) ** 2

            alpha_bars = cos2(torch.arange(N)) / cos2(torch.zeros(1))
            alphas = torch.cat([alpha_bars[:1], alpha_bars[1:] / alpha_bars[:-1]])
            betas = 1 - alphas

            assert torch.allclose(alpha_bars, torch.cumprod(alphas, dim=0))

        return betas, alphas, alpha_bars


    def betas(self, t: torch.LongTensor) -> torch.FloatTensor:
        return self._betas[t]

    def alphas(self, t: torch.LongTensor) -> torch.FloatTensor:
        return self._alphas[t]

    def alpha_bars(self, t: torch.LongTensor) -> torch.FloatTensor:
        return self._alpha_bars[t]

    ### Training / Inference

    def _q_mean(
        self,
        z: torch.FloatTensor, # shape [N, dims] # N being num_samples*num_nodes (num_nodes=1 for the toy dataset)
        t: torch.LongTensor, # shape [N]
    ) -> torch.FloatTensor:
        # TO DO: Implement mean of the distribution q(z_t | z_0)
        # Expected output shape [N, dims]. 
        # Tip: Use the method broadcast_like(sqrt_alpha_bars, z) before multiplying sqrt_alpha_bars with z.
        raise NotImplementedError

    def _q_std(
        self,
        z: torch.FloatTensor, # shape [N, dims] 
        t: torch.LongTensor, # shape [N]
    ) -> torch.FloatTensor:
        # TO DO: std of the distribution q(z_t | z_0)
        # Expected output shape [N, 1]. 
        # Tip: Use the method broadcast_like() to broadcast the computed stds from [N] to [N, 1].
        raise NotImplementedError

    def q_sample(
        self,
        z: torch.FloatTensor,
        t: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Sample from q(z_t | z_0)

        mean = self._q_mean(z, t)
        std = self._q_std(z, t)
        epsilon = torch.randn_like(z)
        z_t = epsilon * std + mean

        return z_t, epsilon

    def _losses(
        self,
        epsilon_pred: torch.FloatTensor, # shape [N, dims]
        epsilon: torch.FloatTensor, # shape [N, dims]
    ) -> torch.FloatTensor:
        # TO DO: Compute the squared error between the predicted and true noise
        # Expected output shape [N]. 
        raise NotImplementedError

    def losses(
        self,
        x: torch.FloatTensor, # shape [N, 3] Where N is num_samples*num_nodes (num_nodes = 1 for toy dataset)
        batch: Optional[torch.LongTensor], # shape [N]
        h: Optional[torch.FloatTensor] = None, # shape [N, 5] (this is None for the toy dataset)
        context: Optional[torch.FloatTensor] = None,
        edge_index: Optional[torch.LongTensor] = None, # shape [2, num_edges]
    ) -> torch.FloatTensor:
        """Compute the uniformly weighted DDPM loss for a batch of samples.

        Args:
            x (torch.FloatTensor): data to diffuse.
            batch (Optional[torch.LongTensor]): batch indices for identifying features belonging to
                the same sample.
            h (Optional[torch.FloatTensor], optional): additional features to diffuse; e.g. one-hot
                representation of the atom label.
            context (Optional[torch.FloatTensor], optional): additional context for conditioning.
            edge_index (Optional[torch.LongTensor], optional): edge indices for the moleculer graph.

        Returns:
            torch.FloatTensor: loss for each sample / atom in the batch.
        """        
        # In case we also have diffuse/denoise h, we concatenate it to x
        if h is not None:
            # TO DO: z will be the concatenation of both x and h on the second axis.
            raise NotImplementedError
        else:
            z = x

        # Sample discrete time between 1 and N-1
        t_discrete = torch.randint(1, self.N, (max(batch) + 1,), device=x.device)
        t_discrete = t_discrete[batch]

        # Compute the loss for a single sample
        z_t, epsilon = self.q_sample(z, t_discrete)

        # Forward pass
        epsilon_pred = self.model(
            z_t,
            t_discrete.float() / self.N,
            edge_index=edge_index,
            batch=batch,
            context=context,
        )

        # Compute the loss for a single sample
        losses = self._losses(epsilon_pred, epsilon)


        # The following ensures that the loss is averaged over the batch.
        # We disabled it, which means for the molecule dataset, the loss is averaged over the
        # number of atoms in the batch. This does not matter for the toy dataset. 
        
        # # Reduce sum if batch is provided
        # if batch is not None:
        #     losses = scatter(losses, batch, dim=0, reduce="sum")

        return losses

    # Sampling

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        context: Optional[torch.FloatTensor] = None,
        edge_index: Optional[torch.LongTensor] = None,
        batch: Optional[torch.LongTensor] = None,
    ) -> np.ndarray:
        # Sample from the prior N(0, I)
        z_t = torch.randn(shape, device=self.device)

        # Iterate from t = N-1 to 1
        for t_discrete in reversed(range(1, self.N)):
            t_discrete = torch.ones(shape[0], device=self.device).long() * t_discrete
            mean = self._p_mean(z_t, t_discrete, edge_index, batch, context)
            std = self._p_std(z_t, t_discrete)
            # sample z_{t-1} | z_t
            epsilon = torch.randn_like(z_t)
            z_t = epsilon * std + mean

        return self.sample_x0_given_x1(z_t, context).cpu().numpy()

    def sample_x0_given_x1(
        self,
        z_1: torch.FloatTensor,
        context: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Approximate it with a Dirac delta distribution.
        # Additional projections can be implemented here.
        return z_1

    def _p_mean(
        self,
        z: torch.FloatTensor, # shape [N, dims]
        t: torch.LongTensor, # shape [N]
        edge_index: Optional[torch.LongTensor] = None, # shape [2, num_edges]
        batch: Optional[torch.LongTensor] = None, # shape [N]
        context: Optional[torch.FloatTensor] = None, 
    ) -> torch.FloatTensor:
        # mean of the distribution p(z_{t-1} | z_t)
        # get beta, alpha, and alpha_bar, all broadcasted to the same shape as x
        beta = broadcast_like(self.betas(t), z) # shape [N, 1]
        alpha = broadcast_like(self.alphas(t), z) # shape [N, 1]
        alpha_bar = broadcast_like(self.alpha_bars(t), z) # shape [N, 1]

        epsilon_pred = self.model(
            z, t.float() / self.N, edge_index=edge_index, batch=batch, context=context
        ) # shape [N, dims]

        # TO DO: Use the ingredients above to compute the mean 
        # Expected shape for mean is [N, dims]

        raise NotImplementedError

    def _p_std(
        self,
        z: torch.FloatTensor,
        t: torch.LongTensor,
    ) -> torch.FloatTensor:
        # std of the distribution p(z_{t-1} | z_t)
        # sigma^2 = beta
        beta = broadcast_like(self.betas(t), z)
        return torch.sqrt(beta)

    @property
    def device(self) -> torch.device:
        return self._betas.device
