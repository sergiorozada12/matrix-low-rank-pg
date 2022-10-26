import torch


class LR(torch.nn.Module):
    def __init__(self, n, m, k, scale=1.0):
        super().__init__()

        L = scale*torch.randn(n, k, dtype=torch.float32, requires_grad=True)
        R = scale*torch.randn(k, m, dtype=torch.float32, requires_grad=True)

        self.L = torch.nn.Parameter(L)
        self.R = torch.nn.Parameter(R)
        
    def forward(self, rows, cols):
        prod = self.L[rows, :] * self.R[:, cols].T
        return torch.sum(prod, dim=-1)