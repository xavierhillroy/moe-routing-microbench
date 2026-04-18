import argparse
import torch
from moe import NaiveMoE, GroupedMoE, GroupedMoECached, build_group_plan


def make_indices(num_tokens, num_experts, regime="uniform", device="cuda"):
    probs = torch.ones(num_experts, device=device, dtype=torch.float32)
    if regime == "moderate":
        probs[0] = 4.0
    elif regime == "heavy":
        probs[0] = 16.0
    elif regime != "uniform":
        raise ValueError(f"Unknown regime: {regime}")
    probs = probs / probs.sum()
    return torch.multinomial(probs, num_tokens, replacement=True)


def run_profile(mode, num_tokens, hidden_dim, ffn_dim, num_experts, regime, warmup, iters):
    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn(num_tokens, hidden_dim, device=device)
    indices = make_indices(num_tokens, num_experts, regime=regime, device=device)

    naive = NaiveMoE(hidden_dim, ffn_dim, num_experts).to(device).eval()
    grouped = GroupedMoE(hidden_dim, ffn_dim, num_experts).to(device).eval()
    grouped_cached = GroupedMoECached(hidden_dim, ffn_dim, num_experts).to(device).eval()

    grouped.load_state_dict(naive.state_dict())
    grouped_cached.load_state_dict(naive.state_dict())

    if mode == "naive":
        model = naive
        args = (x, indices)

    elif mode == "grouped":
        model = grouped
        args = (x, indices)

    elif mode == "grouped_cached":
        plan = build_group_plan(indices, num_experts)
        model = grouped_cached
        args = (x, plan)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(*args)
        torch.cuda.synchronize()

        with torch.cuda.nvtx.range(f"profile.{mode}"):
            for _ in range(iters):
                _ = model(*args)

        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["naive", "grouped", "grouped_cached"], required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--routing", choices=["uniform", "moderate", "heavy"], default="heavy")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    run_profile(
        mode=args.mode,
        num_tokens=args.tokens,
        hidden_dim=args.hidden,
        ffn_dim=args.ffn,
        num_experts=args.experts,
        regime=args.routing,
        warmup=args.warmup,
        iters=args.iters,
    )