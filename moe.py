import torch
import torch.nn as nn

class Expert(nn.Module):
    """
    A single expert FFN.
    Takes input of shape [num_tokens, hidden_dim]
    Returns output of shape [num_tokens, hidden_dim]
    """
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        # nn.Linear is PyTorch's matrix multiplication + bias
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [num_tokens, hidden_dim]
        
        # Step 1: Up-projection
        x = self.fc1(x)   # shape becomes [num_tokens, ffn_dim]
        
        # Step 2: Activation
        x = self.act(x)   # shape remains [num_tokens, ffn_dim]
        
        # Step 3: Down-projection
        x = self.fc2(x)   # shape becomes [num_tokens, hidden_dim]
        
        return x