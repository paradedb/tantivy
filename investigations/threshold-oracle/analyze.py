import json
import sys
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
queries = json.loads((root / "title.json").read_text()) + json.loads(
    (root / "text.json").read_text()
)
summary = []
for query in queries:
    bits = struct.unpack("I", struct.pack("f", query["final_threshold"]))[0]
    relaxed_cutoff = struct.unpack("f", struct.pack("I", bits - 1))[0]
    rows = []
    for run in query["runs"]:
        row = {key: value for key, value in run.items() if key != "timeline"}
        reached = next(
            (
                event
                for event in run["timeline"]
                if event["threshold"] is not None
                and event["threshold"] >= relaxed_cutoff
            ),
            None,
        )
        row["first_final_threshold"] = reached
        rows.append(row)
    summary.append({**{k: v for k, v in query.items() if k != "runs"}, "runs": rows})
    print(f"\n{query['field']}:{query['term']} T*={query['final_threshold']:.8g} "
          f"above={query['docs_above_cutoff']} equal={query['docs_equal_cutoff']}")
    for row in rows:
        print(f"{row['format']:9} {row['policy']:16} "
              f"norms={row['norm_reads']:6} pages={row['norm_pages']:4} "
              f"score={row['scoring_norm_reads']:6} tail={row['tail_bound_norm_reads']:4} "
              f"tailonlypages={row['tail_only_norm_pages']:4} "
              f"candidates={row['candidates']:6} TFreject={row['tf_rejected']:6} "
              f"blocks skip/visit/decode={row['full_blocks_skipped_without_candidates']}/"
              f"{row['full_blocks_visited']}/{row['full_blocks_decoded']}")

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
styles = [
    ("zero", "normal_strict", "Norm-zero, normal", "#1768ac", "-"),
    ("zero", "oracle_address", "Norm-zero, oracle", "#1768ac", "--"),
    ("tf_class", "normal_strict", "TF-class, normal", "#c04c10", "-"),
    ("tf_class", "oracle_address", "TF-class, oracle", "#c04c10", "--"),
]
for index, query in enumerate(queries):
    for fmt, policy, label, color, linestyle in styles:
        run = next(r for r in query["runs"] if r["format"] == fmt and r["policy"] == policy)
        events = run["timeline"]
        x = [event["candidates"] for event in events]
        y = [event["threshold"] if event["threshold"] is not None else 0 for event in events]
        axes[index, 0].step(x, y, where="post", label=label, color=color, linestyle=linestyle)
        axes[index, 1].step(
            x, [event["norm_pages"] for event in events], where="post",
            label=label, color=color, linestyle=linestyle,
        )
    axes[index, 0].axhline(query["final_threshold"], color="#666666", linewidth=0.7, alpha=0.6)
    for col, title in enumerate(["Pruning threshold", "Distinct norm pages touched"]):
        ax = axes[index, col]
        ax.set_title(f"{query['field']}:{query['term']} — {title}")
        ax.set_xlabel("Candidates visited (skipped postings excluded)")
        ax.set_ylabel("BM25 score" if col == 0 else "Logical norm data pages")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
axes[0, 0].legend(loc="lower right", fontsize=8)
fig.suptitle("HN single-term top-10: normal versus an oracle cutoff\n"
             "Exact score/segment/doc-ID tie order preserved; same segment order and postings", fontsize=14)
fig.savefig(root / "threshold-traces.png", dpi=170)
fig.savefig(root / "threshold-traces.svg")
plt.close(fig)
