# -*- coding: utf-8 -*-
"""把 Hugging Face Trainer 的 trainer_state.json 转成表格。"""

import argparse
import csv
import json
from pathlib import Path

TRAIN_FIELDS = ["loss", "learning_rate", "grad_norm"]
EVAL_FIELDS = [
    "eval_loss",
    "eval_accuracy",
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
]
HEADERS = ["step", "epoch", "type"] + TRAIN_FIELDS + EVAL_FIELDS


def find_trainer_state(results_dir):
    matches = sorted(Path(results_dir).rglob("trainer_state.json"))
    if not matches:
        return None

    def score(path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("global_step", -1)
        except Exception:
            return -1

    return max(matches, key=score)


def format_value(key, value):
    if not isinstance(value, (int, float)):
        return value
    if key == "learning_rate":
        return f"{value:.2e}"
    if key == "epoch":
        return f"{value:.2f}"
    if key in ("loss", "grad_norm", "eval_loss", "eval_accuracy"):
        return f"{value:.4f}"
    return value


def build_rows(log_history):
    rows = []
    for entry in log_history:
        is_eval = "eval_accuracy" in entry or "eval_loss" in entry
        row = {
            "step": entry.get("step", ""),
            "epoch": format_value("epoch", entry.get("epoch", "")),
            "type": "eval" if is_eval else "train",
        }
        for key in TRAIN_FIELDS + EVAL_FIELDS:
            value = entry.get(key)
            row[key] = format_value(key, value) if value is not None else ""
        rows.append(row)
    return rows


def to_markdown(rows):
    lines = [
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join(["---"] * len(HEADERS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in HEADERS) + " |")
    return "\n".join(lines)


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="把 Trainer 生成的 trainer_state.json 转成 Markdown / CSV / Excel 表格"
    )
    parser.add_argument("--json", help="直接指定 trainer_state.json 的路径")
    parser.add_argument("--results", default="results", help="results 目录，默认 ./results")
    parser.add_argument("--out-md", default="training_log_table.md", help="Markdown 输出文件")
    parser.add_argument("--out-csv", default="training_log_table.csv", help="CSV 输出文件")
    parser.add_argument("--excel", action="store_true", help="额外生成 xlsx，需要 openpyxl")
    args = parser.parse_args()

    if args.json:
        state_path = Path(args.json)
    else:
        state_path = find_trainer_state(args.results)

    if state_path is None:
        raise SystemExit(
            f"没有找到 trainer_state.json，请检查目录：{Path(args.results).resolve()}"
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    if not log_history:
        raise SystemExit("trainer_state.json 里没有 log_history，说明训练还没有产生日志")

    rows = build_rows(log_history)
    print(to_markdown(rows))

    md_path = Path(args.out_md)
    md_path.write_text(to_markdown(rows), encoding="utf-8")
    write_csv(rows, Path(args.out_csv))

    print(f"\n已保存：{md_path.resolve()}")
    print(f"已保存：{Path(args.out_csv).resolve()}")

    if state.get("best_metric") is not None:
        print(
            "最佳结果："
            f"{state['best_metric']} "
            f"(step {state.get('best_global_step', '')})"
        )

    if args.excel:
        try:
            import pandas as pd

            excel_path = Path(args.out_csv).with_suffix(".xlsx")
            pd.DataFrame(rows).to_excel(excel_path, index=False)
            print(f"已保存：{excel_path.resolve()}")
        except ImportError:
            print("没有安装 openpyxl，跳过 Excel；可执行 pip install openpyxl")


if __name__ == "__main__":
    main()
