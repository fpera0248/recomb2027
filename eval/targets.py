"""Which (label, finer type) pairs count as hidden rare subtypes in a natural pool.

A target is a pair (visible label L, finer annotation T) with T != L, T not a placeholder,
at least 10 cells, and under 10% of the label. The union of all target cells in a pool is the
set a method is asked to rank first.
"""
import pandas as pd

BAD = {"None", "nan", "Unknown"}


def targets(df):
    """df has label and true_type per cell -> [(label, type, n_cells)]."""
    nl = df.label.value_counts()
    out = []
    for (lab, typ), n in df.groupby(["label", "true_type"]).size().items():
        if typ != lab and typ not in BAD and n >= 10 and n / nl[lab] < 0.10:
            out.append((lab, typ, int(n)))
    return out


if __name__ == "__main__":
    df = pd.DataFrame({"label": ["a"] * 200 + ["b"] * 50, "true_type": ["a"] * 185 + ["x"] * 15 + ["b"] * 40 + ["y"] * 10})
    assert targets(df) == [("a", "x", 15)]       # y is 20% of b, not rare
