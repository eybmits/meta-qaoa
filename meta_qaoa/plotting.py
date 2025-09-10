
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_curve(df_list: List[Tuple[str, pd.DataFrame]], title: str, out_png: str):
    plt.figure()
    drew = False
    markers = {
        "meta-qaoa": "o",
        "opt-meta-qaoa": "s",
        "meta-classical": "^",
        "opt-meta-classical": "v",
    }
    for name, df in df_list:
        if len(df) == 0:
            continue
        x = np.asarray(df["lambda"].values)
        m = np.asarray(df["median"].values)
        ql = np.asarray(df["q25"].values)
        qu = np.asarray(df["q75"].values)
        o = np.argsort(x)
        x, m, ql, qu = x[o], m[o], ql[o], qu[o]
        plt.plot(x, m, marker=markers.get(name, "o"), linestyle="-", label=name)
        plt.fill_between(x, ql, qu, alpha=0.2)
        drew = True
    plt.axhline(0.98, linestyle=":", label="98% Ziel (E/E*)")
    if drew:
        plt.xlabel("λ"); plt.ylabel("E(λ)/E*(λ) – Median, IQR")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=160)
        print(f"[Saved] {out_png}")
        plt.close()
