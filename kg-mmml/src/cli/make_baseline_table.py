# src/cli/make_baseline_table.py
import argparse, csv, json, pathlib

def parse_labeled_input(s: str):
    """
    Accepts 'path[:label]'. If ':label' is omitted, we'll use the JSON's 'mode' field.
    """
    if ":" in s:
        path, label = s.split(":", 1)
    else:
        path, label = s, ""
    return path.strip(), label.strip()

def load_metrics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fmt(x, ndp=4):
    try:
        return f"{float(x):.{ndp}f}"
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="List of metrics JSONs as 'path[:label]'. "
                         "If label omitted, uses JSON 'mode' field.")
    ap.add_argument("--out", default="reports/tables/baseline_vs_kge.csv",
                    help="Output CSV path")
    ap.add_argument("--notes", nargs="*", default=[],
                    help="Optional notes to append per row, same order as inputs")
    args = ap.parse_args()

    rows = []
    for i, spec in enumerate(args.inputs):
        path, label = parse_labeled_input(spec)
        m = load_metrics(path)

        mode = label or m.get("mode", "")
        row = {
            "mode": mode,
            "micro_f1": fmt(m.get("micro_f1")),
            "macro_f1": fmt(m.get("macro_f1")),
            "n_docs_train": m.get("n_docs_train", ""),
            "n_docs_test": m.get("n_docs_test", ""),
            "notes": args.notes[i] if i < len(args.notes) else m.get("notes", "")
        }
        rows.append(row)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode","micro_f1","macro_f1","n_docs_train","n_docs_test","notes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[table] wrote {outp} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
