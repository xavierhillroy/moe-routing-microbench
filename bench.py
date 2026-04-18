import os
import torch
import pandas as pd
from moe import NaiveMoE, GroupedMoE, GroupedMoECached, build_group_plan


def benchmark_moe(model, *args, num_runs=100, warmup_runs=10):
    model.eval()

    with torch.inference_mode():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(*args)
        torch.cuda.synchronize()

        # Timed runs
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            _ = model(*args)
        end_event.record()

        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_runs  # ms


def make_indices(num_tokens, num_experts, regime="uniform", device="cuda"):
    probs = torch.ones(num_experts, device=device, dtype=torch.float32)

    if regime == "uniform":
        pass
    elif regime == "moderate":
        probs[0] = 4.0
    elif regime == "heavy":
        probs[0] = 16.0
    else:
        raise ValueError(f"Unknown regime: {regime}")

    probs = probs / probs.sum()
    return torch.multinomial(probs, num_tokens, replacement=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    os.makedirs("results", exist_ok=True)

    device = "cuda"
    torch.manual_seed(0)

    # Start with a routing-sensitive configuration.
    # Your previous ffn_dim=4096 made compute dominate too much.
    hidden_dim = 256
    ffn_dim = 512
    num_experts = 64

    token_counts = [512, 2048, 8192, 16384, 32768]
    regimes = ["uniform", "heavy"]

    results = []

    for regime in regimes:
        print(f"\n=== Routing regime: {regime} ===")

        for num_tokens in token_counts:
            x = torch.randn(num_tokens, hidden_dim, device=device)
            indices = make_indices(num_tokens, num_experts, regime=regime, device=device)

            naive = NaiveMoE(hidden_dim, ffn_dim, num_experts).to(device)
            grouped = GroupedMoE(hidden_dim, ffn_dim, num_experts).to(device)
            grouped_cached = GroupedMoECached(hidden_dim, ffn_dim, num_experts).to(device)

            # Make all three mathematically identical
            grouped.load_state_dict(naive.state_dict())
            grouped_cached.load_state_dict(naive.state_dict())

            # Build cached grouping plan ONCE, outside timed region
            plan = build_group_plan(indices, num_experts)

            # Correctness check
            with torch.inference_mode():
                out_naive = naive(x, indices)
                out_grouped = grouped(x, indices)
                out_grouped_cached = grouped_cached(x, plan)

            if not torch.allclose(out_naive, out_grouped, atol=1e-5):
                raise RuntimeError(f"NaiveMoE and GroupedMoE mismatch for tokens={num_tokens}, regime={regime}")

            if not torch.allclose(out_naive, out_grouped_cached, atol=1e-5):
                raise RuntimeError(f"NaiveMoE and GroupedMoECached mismatch for tokens={num_tokens}, regime={regime}")

            t_naive = benchmark_moe(naive, x, indices)
            t_grouped = benchmark_moe(grouped, x, indices)
            t_grouped_cached = benchmark_moe(grouped_cached, x, plan)

            speedup_grouped = t_naive / t_grouped
            speedup_cached = t_naive / t_grouped_cached

            counts = torch.bincount(indices, minlength=num_experts).cpu().tolist()

            print(
                f"Tokens: {num_tokens:<6} | "
                f"Naive: {t_naive:.3f} ms | "
                f"Grouped: {t_grouped:.3f} ms | "
                f"GroupedCached: {t_grouped_cached:.3f} ms | "
                f"Naive/Grouped: {speedup_grouped:.2f}x | "
                f"Naive/Cached: {speedup_cached:.2f}x"
            )
            print(f"Expert counts: {counts}")

            results.append({
                "routing_regime": regime,
                "tokens": num_tokens,
                "hidden_dim": hidden_dim,
                "ffn_dim": ffn_dim,
                "num_experts": num_experts,
                "naive_ms": t_naive,
                "grouped_ms": t_grouped,
                "grouped_cached_ms": t_grouped_cached,
                "naive_over_grouped": speedup_grouped,
                "naive_over_grouped_cached": speedup_cached,
                "expert_counts": str(counts),
            })

    df = pd.DataFrame(results)
    df.to_csv("results/benchmark_results.csv", index=False)

    print("\nSaved results to results/benchmark_results.csv")