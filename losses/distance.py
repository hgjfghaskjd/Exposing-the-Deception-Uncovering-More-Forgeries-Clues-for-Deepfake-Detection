import torch
import torch.nn as nn

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff

import torch

class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.01, max_iter=100, reduction='mean',device='cuda:0'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.device=device
        
    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        C = C.to(self.device)
        n_points = x.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, n_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / n_points).squeeze()
        nu = torch.empty(batch_size, n_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / n_points).squeeze()

        u = torch.zeros_like(mu)
        u = u.to(self.device)
        v = torch.zeros_like(nu)
        v = v.to(self.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8).to(self.device) - self.lse(self.M(C, u, v))).to(self.device) + u
            v = self.eps * (torch.log(nu + 1e-8).to(self.device) - self.lse(self.M(C, u, v).transpose(-2, -1))).to(self.device) + v
            err = (u - u1).abs().sum(-1).mean()
            err = err.to(self.device)

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def lse(A):
        "log-sum-exp"
        # add 10^-6 to prevent NaN
        result = torch.log(torch.exp(A).sum(-1) + 1e-6)
        return result

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1