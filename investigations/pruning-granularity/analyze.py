import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
queries = json.loads((root / "title.json").read_text()) + json.loads((root / "text.json").read_text())
previous = json.loads((root.parent / "threshold-oracle/title.json").read_text()) + json.loads(
    (root.parent / "threshold-oracle/text.json").read_text()
)
summary = []
for query in queries:
    old = next(q for q in previous if (q["field"], q["term"]) == (query["field"], query["term"]))
    rows = [{k: v for k, v in run.items() if k != "entered_groups"} for run in query["runs"]]
    for row in rows:
        if row.get("bound_mode") == "stored":
            old_run = next(r for r in old["runs"] if r["format"] == row["format"] and r["policy"] == "oracle_address")
            for key in ["norm_reads", "norm_pages", "candidates", "tf_rejected", "full_blocks_visited"]:
                assert row[key] == old_run[key], (query["field"], query["term"], row["format"], key)
        if row.get("group_size") == 1:
            assert row["norm_reads"] == 10
    blocks = query["blocks"]
    counts = {
        "equal": sum(b["stored"] == b["exact"] for b in blocks),
        "over": sum(b["stored"] > b["exact"] for b in blocks),
        "under": sum(b["stored"] < b["exact"] for b in blocks),
    }
    print(f"\n{query['field']}:{query['term']} blocks={len(blocks)} bounds={counts}")
    for row in rows:
        print(f"{row['format']:9} {str(row.get('bound_mode', row.get('group_size'))):11} "
              f"norms={row['norm_reads']:5} pages={row['norm_pages']:4} "
              f"candidates={row['candidates']:5} TFreject={row['tf_rejected']:5}")
    summary.append({**{k: v for k, v in query.items() if k not in ["runs", "blocks"]},
                    "bound_comparison": counts, "underestimated_blocks": [b for b in blocks if b["stored"] < b["exact"]],
                    "runs": rows})
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14, 5.1), layout="constrained")
labels = ["Stored\n128", "Exact\nfull only", "Exact 128\n+ tails", "64", "32", "16", "8", "1\nideal"]
for ax, query in zip(axes, queries):
    for fmt, label, color in [("zero", "Norm-zero TF filter", "#1768ac"), ("tf_class", "TF-class minima (128)", "#c04c10")]:
        rows = [r for r in query["runs"] if r["format"] == fmt]
        points = [next(r for r in rows if r.get("bound_mode") == mode) for mode in ["stored", "exact_full", "exact_all"]]
        points += [next(r for r in rows if r.get("group_size") == size) for size in [64, 32, 16, 8, 1]]
        y = [r["norm_pages"] for r in points]
        ax.plot(range(len(y)), y, marker="o", color=color, label=label)
        if fmt == "zero":
            for x, value in enumerate(y):
                ax.annotate(str(value), (x, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=color)
    ax.set_title(f"{query['field']}:{query['term']}")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("Distinct logical norm data pages")
    ax.set_xlabel("Pruning group size (postings)")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    ax.grid(alpha=0.15)
axes[0].legend(fontsize=8)
fig.suptitle("HN single-term top-10: exact bounds at finer granularity\n"
             "Fixed oracle cutoff and tie order; first 3 points native, smaller groups simulated", fontsize=13)
fig.savefig(root / "granularity.png", dpi=170)
fig.savefig(root / "granularity.svg")
plt.close(fig)
