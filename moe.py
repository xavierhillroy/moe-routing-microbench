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

class NaiveMoE(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        # nn.ModuleList registers these experts so PyTorch knows to move them 
        # to the GPU when you call .cuda() on the parent module.
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        x: [num_tokens, hidden_dim]
        expert_indices: [num_tokens] (1D tensor of ints mapping token -> expert)
        """
        # Allocate an empty output tensor in VRAM (our scatter destination)
        output = torch.zeros_like(x)
        #Naive batching -
        for i, expert in enumerate(self.experts):
            # 1. Find the 1D indices of tokens assigned to expert i
            token_idx = (expert_indices == i).nonzero(as_tuple=True)[0]
            
            # 2. Check if this expert got any tokens (skip if empty)
            if token_idx.numel() == 0:
                continue
                
            # 3. GATHER (The Bottleneck): Pull scattered tokens into a contiguous block
            expert_input = x[token_idx]
            
            # 4. COMPUTE: Fast dense matrix multiplication
            expert_output = expert(expert_input)
            
            # 5. SCATTER (The Bottleneck): Write scattered tokens back to output
            output[token_idx] = expert_output
            
        return output

class GroupedMoE(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        # STEP 1: Sort the indices to group tokens by expert
        # sorted_expert_indices: e.g., [0, 0, 1, 1, 1, 2, ...]
        # sort_order: the original row numbers, used so we can unsort later
        sorted_expert_indices, sort_order = torch.sort(expert_indices)
        
        # STEP 2: Rearrange the input tensor ONCE (One coalesced gather)
        grouped_x = x[sort_order]
        
        # Allocate output tensor for the grouped results
        grouped_output = torch.zeros_like(grouped_x)
        
        # STEP 3: Count how many tokens belong to each expert
        # bincount returns an array of counts: e.g., [2, 3, 1] means Expert 0 got 2 tokens.
        # .cpu().tolist() moves it to standard Python integers so we can loop over it safely.
        tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=self.num_experts).cpu().tolist()
        
        # STEP 4: Process sequentially using slices (Zero-copy!)
        current_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
                
            # SLICE: This does not copy memory, it just offsets a C++ pointer
            expert_input = grouped_x[current_idx : current_idx + num_tokens]
            
            # COMPUTE
            expert_output = self.experts[i](expert_input)
            
            # SLICE WRITE
            grouped_output[current_idx : current_idx + num_tokens] = expert_output
            
            current_idx += num_tokens
            
        # STEP 5: Restore original order (One coalesced scatter)
        # We use the sort_order array as the index destination
        output = torch.empty_like(grouped_output)
        output[sort_order] = grouped_output
        
        return output

if __name__ == "__main__":
    hidden_dim = 512
    ffn_dim = 2048
    num_tokens = 128
    num_experts = 4

    # 1. Instantiate both models
    moe_naive = NaiveMoE(hidden_dim, ffn_dim, num_experts).cuda()
    moe_grouped = GroupedMoE(hidden_dim, ffn_dim, num_experts).cuda()

    # CRITICAL: Copy the random weights from Naive to Grouped so they are mathematically identical
    moe_grouped.load_state_dict(moe_naive.state_dict())

    # 2. Create data
    x = torch.randn(num_tokens, hidden_dim).cuda()
    expert_indices = torch.randint(0, num_experts, (num_tokens,)).cuda()

    # 3. Run both
    out_naive = moe_naive(x, expert_indices)
    out_grouped = moe_grouped(x, expert_indices)

    # 4. Verify Correctness
    # torch.allclose checks if two floating-point tensors are equal within a tiny margin of error
    is_correct = torch.allclose(out_naive, out_grouped, atol=1e-5)
    
    print(f"Naive Output Shape: {out_naive.shape}")
    print(f"Grouped Output Shape: {out_grouped.shape}")
    
    if is_correct:
        print("SUCCESS: GroupedMoE and NaiveMoE produce mathematically identical outputs!")
    else:
        print("ERROR: Outputs do not match. We broke the math.")