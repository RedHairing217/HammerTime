"""
start.py
========
HammerTime terminal interface.

Usage
-----
    python src/start.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "src" else Path(__file__).parent

CLASS_NAMES = ["hammer", "axe", "wrench", "pickaxe"]

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def clear() -> None:
    print("\n" + "=" * 52)


def header(title: str) -> None:
    clear()
    print(f"  HammerTime  |  {title}")
    print("=" * 52)


def menu(options: list[str]) -> None:
    print()
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print()


def prompt(msg: str = "Select") -> str:
    return input(f"  {msg}: ").strip().lower()


def confirm(msg: str) -> bool:
    print(f"\n  {msg}")
    print("  [y] Confirm   [b] Back")
    r = prompt()
    return r == "y"


def select_option(options: list[str], title: str) -> int | None:
    """Show a menu and return 1-based index, or None if back."""
    header(title)
    menu(options + ["Back"])
    while True:
        r = prompt()
        if r == str(len(options) + 1) or r == "b":
            return None
        try:
            i = int(r)
            if 1 <= i <= len(options):
                return i
        except ValueError:
            pass
        print("  Invalid selection.")


def select_class(title: str) -> str | None:
    """Class selection menu. Returns class name or None if back."""
    header(title)
    menu(CLASS_NAMES + ["Back"])
    while True:
        r = prompt()
        if r == str(len(CLASS_NAMES) + 1) or r == "b":
            return None
        try:
            i = int(r)
            if 1 <= i <= len(CLASS_NAMES):
                return CLASS_NAMES[i - 1]
        except ValueError:
            pass
        print("  Invalid selection.")


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> None:
    print()
    subprocess.run([sys.executable] + cmd)
    input("\n  Press Enter to continue.")


def run_prepare(target_class: str | None = None) -> None:
    cmd = [str(ROOT / "src/shared/prepare_dataset.py")]
    if target_class:
        cmd += ["--target_class", target_class]
    print("\n  Preparing dataset...")
    subprocess.run([sys.executable] + cmd, cwd=ROOT)


def run_train(mode: str, target_class: str | None = None) -> None:
    run_prepare(target_class)

    header("Dashboard")
    print("\n  Launch dashboard publicly (ngrok) or local only?\n")
    print("  [1] Local only  (http://localhost:7842)")
    print("  [2] Public URL  (ngrok -- requires ngrok installed)")
    print()
    r = prompt()

    dashboard = str(ROOT / "src/dashboard.py")
    cmd       = [sys.executable, dashboard, "--mode", mode]
    if target_class:
        cmd += ["--target_class", target_class]
    if r == "2":
        cmd += ["--public"]

    print("\n  Launching training and dashboard...")
    subprocess.run(cmd, cwd=ROOT)
    input("\n  Training complete. Press Enter to continue.")


def run_evaluate(mode: str) -> None:
    script = "src/binary/evaluate_b.py" if mode == "binary" else "src/multi/evaluate_m.py"
    run([str(ROOT / script), "--split", "test"])


def run_failures(mode: str, class_id: int | None = None) -> None:
    script = "src/binary/analyze_failures_b.py" if mode == "binary" else "src/multi/analyze_failures_m.py"
    cmd = [str(ROOT / script)]
    if mode == "multi" and class_id is not None:
        cmd += ["--class_id", str(class_id)]
    run(cmd)


def run_false_positives(mode: str, class_id: int | None = None) -> None:
    script = "src/binary/analyze_false_positives_b.py" if mode == "binary" else "src/multi/analyze_false_positives_m.py"
    cmd = [str(ROOT / script)]
    if mode == "multi" and class_id is not None:
        cmd += ["--class_id", str(class_id)]
    run(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Menus
# ─────────────────────────────────────────────────────────────────────────────

def menu_single_class() -> None:
    while True:
        choice = select_option(
            ["Train", "Evaluate", "Analyze failures", "Analyze false positives"],
            "Single-class detector"
        )
        if choice is None:
            return

        action_label = {1: "Train", 2: "Evaluate", 3: "Analyze failures", 4: "Analyze false positives"}[choice]
        target_class = select_class(action_label)
        if target_class is None:
            continue

        if choice == 1:
            if confirm(f"Ready to train single-class detector on [{target_class}]?"):
                run_train("binary", target_class)

        elif choice == 2:
            if confirm(f"Evaluate single-class [{target_class}] on held-out test set?"):
                run_evaluate("binary")

        elif choice == 3:
            if confirm(f"Analyze failures for single-class [{target_class}]?"):
                run_failures("binary")

        elif choice == 4:
            if confirm(f"Analyze false positives for single-class [{target_class}]?"):
                run_false_positives("binary")


def menu_multi_class() -> None:
    while True:
        choice = select_option(
            ["Train", "Evaluate", "Analyze failures", "Analyze false positives"],
            "Multi-class detector"
        )
        if choice is None:
            return

        if choice == 1:
            if confirm("Ready to train multi-class detector?"):
                run_train("multi")

        elif choice == 2:
            if confirm("Evaluate multi-class detector on held-out test set?"):
                run_evaluate("multi")

        elif choice in (3, 4):
            header("Filter by class")
            menu(CLASS_NAMES + ["All classes", "Back"])
            while True:
                r = prompt()
                if r == str(len(CLASS_NAMES) + 2) or r == "b":
                    break
                try:
                    i = int(r)
                    if i == len(CLASS_NAMES) + 1:
                        class_id = None
                        class_label = "all classes"
                    elif 1 <= i <= len(CLASS_NAMES):
                        class_id = i - 1
                        class_label = CLASS_NAMES[class_id]
                    else:
                        print("  Invalid selection.")
                        continue
                except ValueError:
                    print("  Invalid selection.")
                    continue

                label = "failures" if choice == 3 else "false positives"
                if confirm(f"Analyze {label} for [{class_label}]?"):
                    if choice == 3:
                        run_failures("multi", class_id)
                    else:
                        run_false_positives("multi", class_id)
                break


def menu_split() -> None:
    header("Split training data")
    print("\n  Splits data/ into dataset/ at 70/15/15.")
    print("  Test set is locked after first run.")
    if confirm("Run 70/15/15 split now?"):
        run_prepare()


def reopen_dashboard() -> None:
    header("Re-open dashboard")
    print("\n  Which pipeline?\n")
    print("  [1] Single-class detector")
    print("  [2] Multi-class detector")
    print()
    mode_r = prompt()
    mode   = "binary" if mode_r == "1" else "multi"

    print("\n  Local or public?\n")
    print("  [1] Local only  (http://localhost:7842)")
    print("  [2] Public URL  (ngrok)")
    print()
    r = prompt()

    dashboard = str(ROOT / "src/dashboard.py")
    cmd       = [sys.executable, dashboard, "--mode", mode, "--watch_only"]
    if r == "2":
        cmd += ["--public"]

    print("\n  Opening dashboard...")
    subprocess.run(cmd, cwd=ROOT)
    input("\n  Press Enter to continue.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    while True:
        header("Main menu")
        menu([
            "Single-class detector",
            "Multi-class detector",
            "Re-open dashboard",
            "Split training data",
            "Settings",
            "Exit",
        ])
        r = prompt()

        if r == "1":
            menu_single_class()
        elif r == "2":
            menu_multi_class()
        elif r == "3":
            reopen_dashboard()
        elif r == "4":
            menu_split()
        elif r == "5":
            header("Settings")
            print("\n  Settings not yet configured.")
            input("\n  Press Enter to continue.")
        elif r in ("6", "q", "exit"):
            print("\n  Exiting.\n")
            sys.exit(0)
        else:
            print("  Invalid selection.")


if __name__ == "__main__":
    main()
