import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_latency(df: pd.DataFrame, regime: str, outdir: str) -> str:
    sub = df[df["routing_regime"] == regime].copy()
    sub = sub.sort_values("tokens")

    plt.figure(figsize=(8, 5))
    plt.plot(sub["tokens"], sub["naive_ms"], marker="o", label="Naive")
    plt.plot(sub["tokens"], sub["grouped_ms"], marker="o", label="Grouped")
    plt.plot(sub["tokens"], sub["grouped_cached_ms"], marker="o", label="GroupedCached")

    plt.xlabel("Number of tokens")
    plt.ylabel("Latency (ms)")
    plt.title(f"MoE latency vs tokens ({regime} routing)")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"latency_{regime}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_speedup(df: pd.DataFrame, regime: str, outdir: str) -> str:
    sub = df[df["routing_regime"] == regime].copy()
    sub = sub.sort_values("tokens")

    plt.figure(figsize=(8, 5))
    plt.plot(sub["tokens"], sub["naive_over_grouped"], marker="o", label="Naive / Grouped")
    plt.plot(sub["tokens"], sub["naive_over_grouped_cached"], marker="o", label="Naive / GroupedCached")

    plt.xlabel("Number of tokens")
    plt.ylabel("Speedup (x)")
    plt.title(f"MoE speedup vs tokens ({regime} routing)")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"speedup_{regime}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_summary_bar(df: pd.DataFrame, outdir: str) -> str:
    rows = []
    for regime in sorted(df["routing_regime"].unique()):
        sub = df[df["routing_regime"] == regime].copy()
        sub = sub.sort_values("tokens")
        last = sub.iloc[-1]
        rows.append({
            "routing_regime": regime,
            "tokens": int(last["tokens"]),
            "naive_ms": float(last["naive_ms"]),
            "grouped_ms": float(last["grouped_ms"]),
            "grouped_cached_ms": float(last["grouped_cached_ms"]),
        })

    summary = pd.DataFrame(rows)

    x = range(len(summary))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar([i - width for i in x], summary["naive_ms"], width=width, label="Naive")
    plt.bar(x, summary["grouped_ms"], width=width, label="Grouped")
    plt.bar([i + width for i in x], summary["grouped_cached_ms"], width=width, label="GroupedCached")

    labels = [
        f'{row["routing_regime"]}\n{row["tokens"]} tokens'
        for _, row in summary.iterrows()
    ]

    plt.xticks(list(x), labels)
    plt.ylabel("Latency (ms)")
    plt.title("Latency at largest token count per routing regime")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, "latency_summary_bar.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/benchmark_results.csv")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required_cols = {
        "routing_regime",
        "tokens",
        "naive_ms",
        "grouped_ms",
        "grouped_cached_ms",
        "naive_over_grouped",
        "naive_over_grouped_cached",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    generated = []

    for regime in sorted(df["routing_regime"].unique()):
        generated.append(plot_latency(df, regime, args.outdir))
        generated.append(plot_speedup(df, regime, args.outdir))

    generated.append(plot_summary_bar(df, args.outdir))

    print("Generated figures:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()