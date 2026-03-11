"""Matplotlib helpers for experiment charts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"[plots] Matplotlib unavailable: {exc}")
        print("[plots] Install with: pip install matplotlib")
        return None


def save_bar_chart(
    *,
    output_path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    x_label: str,
    y_label: str,
    color: str = "#4C78A8",
    rotate_x: int = 0,
) -> bool:
    plt = _get_plt()
    if plt is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(list(labels), list(values), color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[plots] saved: {output_path}")
    return True


def save_line_chart(
    *,
    output_path: Path,
    title: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_label: str,
    y_label: str,
    color: str = "#F58518",
    marker: str = "o",
) -> bool:
    plt = _get_plt()
    if plt is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(list(x_values), list(y_values), color=color, marker=marker, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[plots] saved: {output_path}")
    return True


def save_multi_line_chart(
    *,
    output_path: Path,
    title: str,
    x_values: Sequence[float],
    series: list[tuple[str, Sequence[float], str]],
    x_label: str,
    y_label: str,
) -> bool:
    """series item format: (name, y_values, color)."""
    plt = _get_plt()
    if plt is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, y_values, color in series:
        ax.plot(list(x_values), list(y_values), marker="o", linewidth=2, label=name, color=color)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[plots] saved: {output_path}")
    return True
