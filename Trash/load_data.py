"""Load demo admission/discharge lists from ward_analysis2.py without editing it."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

TARGET_VARS = ("admissions", "discharges", "post_admissions", "post_discharges")


class DemoDataNotFoundError(RuntimeError):
    """Raised when the demo data block cannot be found or parsed."""



def _is_main_guard(node: ast.If) -> bool:
    """Return True when the node is: if __name__ == '__main__':"""
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comp = test.comparators[0]
    return isinstance(comp, ast.Constant) and comp.value == "__main__"



def load_demo_data(source_file: str | Path = "ward_analysis2.py") -> Dict[str, List[int]]:
    """
    Parse the __main__ block of ward_analysis2.py and extract the demo lists.

    This lets you keep ward_analysis2.py as the single editable data source.
    """
    source_path = Path(source_file)
    if not source_path.exists():
        raise FileNotFoundError(f"Cannot find source file: {source_path}")

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    data: Dict[str, List[int]] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard(node):
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    continue
                var_name = stmt.targets[0].id
                if var_name not in TARGET_VARS:
                    continue
                try:
                    value = ast.literal_eval(stmt.value)
                except Exception as exc:  # pragma: no cover - defensive
                    raise DemoDataNotFoundError(
                        f"Failed to parse variable '{var_name}' in {source_path}"
                    ) from exc
                if not isinstance(value, (list, tuple)):
                    raise DemoDataNotFoundError(
                        f"Variable '{var_name}' in {source_path} is not a list/tuple."
                    )
                data[var_name] = [int(x) for x in value]
            break

    missing = [name for name in TARGET_VARS if name not in data]
    if missing:
        raise DemoDataNotFoundError(
            f"Missing variables in {source_path}: {', '.join(missing)}"
        )

    return data



def get_dataset(dataset: str = "pre", source_file: str | Path = "ward_analysis2.py") -> Dict[str, List[int]]:
    """Return the requested 7-day dataset."""
    dataset = dataset.lower().strip()
    data = load_demo_data(source_file)

    if dataset == "pre":
        admissions = data["admissions"]
        discharges = data["discharges"]
    elif dataset == "post":
        admissions = data["post_admissions"]
        discharges = data["post_discharges"]
    else:
        raise ValueError("dataset must be 'pre' or 'post'")

    if len(admissions) != 7 or len(discharges) != 7:
        raise ValueError("Each dataset must contain exactly 7 days of data.")

    return {
        "dataset": dataset,
        "admissions": admissions,
        "discharges": discharges,
    }


if __name__ == "__main__":
    demo = get_dataset("pre")
    print("Loaded dataset:")
    print(demo)
