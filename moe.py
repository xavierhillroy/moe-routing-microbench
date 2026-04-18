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
        output = torch.empty_like(x)

        with torch.cuda.nvtx.range("naive.expert_loop"):
            for i, expert in enumerate(self.experts):
                token_idx = (expert_indices == i).nonzero(as_tuple=True)[0]
                if token_idx.numel() == 0:
                    continue

                expert_input = x[token_idx]
                expert_output = expert(expert_input)
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
        with torch.cuda.nvtx.range("grouped.sort"):
            sorted_expert_indices, sort_order = torch.sort(expert_indices)

        with torch.cuda.nvtx.range("grouped.reorder_input"):
            grouped_x = x[sort_order]

        grouped_output = torch.empty_like(grouped_x)

        with torch.cuda.nvtx.range("grouped.counts_to_cpu"):
            tokens_per_expert = torch.bincount(
                sorted_expert_indices, minlength=self.num_experts
            ).cpu().tolist()

        current_idx = 0
        with torch.cuda.nvtx.range("grouped.expert_loop"):
            for i, num_tokens in enumerate(tokens_per_expert):
                if num_tokens == 0:
                    continue
                expert_input = grouped_x[current_idx: current_idx + num_tokens]
                expert_output = self.experts[i](expert_input)
                grouped_output[current_idx: current_idx + num_tokens] = expert_output
                current_idx += num_tokens

        with torch.cuda.nvtx.range("grouped.restore_output"):
            output = torch.empty_like(grouped_output)
            output[sort_order] = grouped_output

        return output
def build_group_plan(expert_indices: torch.Tensor, num_experts: int):
    with torch.cuda.nvtx.range("group_plan.sort"):
        sorted_experts, sort_order = torch.sort(expert_indices)

    with torch.cuda.nvtx.range("group_plan.bincount"):
        counts = torch.bincount(sorted_experts, minlength=num_experts)

    with torch.cuda.nvtx.range("group_plan.offsets"):
        offsets = torch.zeros(
            num_experts + 1,
            device=expert_indices.device,
            dtype=torch.long
        )
        offsets[1:] = counts.cumsum(0)

    with torch.cuda.nvtx.range("group_plan.restore_order"):
        restore_order = torch.empty_like(sort_order)
        restore_order[sort_order] = torch.arange(
            sort_order.numel(), device=expert_indices.device
        )

    with torch.cuda.nvtx.range("group_plan.to_cpu"):
        offsets_cpu = offsets.cpu().tolist()

    return {
        "sort_order": sort_order,
        "restore_order": restore_order,
        "offsets_cpu": offsets_cpu,
    }


class GroupedMoECached(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, plan) -> torch.Tensor:
        sort_order = plan["sort_order"]
        restore_order = plan["restore_order"]
        offsets = plan["offsets_cpu"]

        with torch.cuda.nvtx.range("grouped_cached.reorder_input"):
            grouped_x = x[sort_order]

        grouped_output = torch.empty_like(grouped_x)

        with torch.cuda.nvtx.range("grouped_cached.expert_loop"):
            for i in range(self.num_experts):
                start = offsets[i]
                end = offsets[i + 1]
                if start == end:
                    continue

                expert_input = grouped_x[start:end]
                grouped_output[start:end] = self.experts[i](expert_input)

        with torch.cuda.nvtx.range("grouped_cached.restore_output"):
            return grouped_output[restore_order]
