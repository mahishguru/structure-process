"""Build the canonical labels.csv: composition + process parameters per condition.

Sources:
- Composition: data_labels.xlsx (wt% per alloy; nominal/measured cast compositions).
- Process:     database_24_06 folder names  <alloy>/<T>_<v>  with T in deg C and
               v in mm/s (extrusion ram speed).

Output columns:
    condition_id, alloy, base_alloy, heat_treated, alloy_class_idx,
    Al, Zn, Mn, Ce, Gd, Ca, Nd, Y  (wt%),
    T_ext (deg C), v_ext (mm/s)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import openpyxl

from icme_mg import ELEMENTS

# xlsx alloy-name quirks -> canonical base alloy name
_XLSX_NAME_FIXES = {
    "Mg-10Gd-1Mn)": "Mg-10Gd-1Mn",
    "ZNd10 = Mg-Zn-Nd": "ZNd10",
    "ZX10 = Mg-Zn-Ca": "ZX10",
}

# xlsx element columns of interest (xlsx header -> canonical element symbol)
_XLSX_ELEMENTS = {"Al": "Al", "Zn": "Zn", "Mn": "Mn", "Ce": "Ce",
                  "Gd": "Gd", "Ca": "Ca", "Nd": "Nd", "Y": "Y"}

_COND_RE = re.compile(r"^(?P<T>\d+(?:\.\d+)?)_(?P<v>\d+(?:\.\d+)?)$")


def read_composition_table(xlsx_path: str | Path) -> pd.DataFrame:
    """Parse data_labels.xlsx into a DataFrame indexed by base alloy name."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    # Header row is the one starting with 'alloy'
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "alloy")
    header = [str(c).strip() if c is not None else "" for c in rows[header_idx]]
    records = []
    for row in rows[header_idx + 1:]:
        name = row[0]
        if not isinstance(name, str) or not name.strip():
            continue
        name = _XLSX_NAME_FIXES.get(name.strip(), name.strip())
        rec = {"base_alloy": name}
        valid = False
        for col, sym in _XLSX_ELEMENTS.items():
            if col in header:
                val = row[header.index(col)]
                if val is not None:
                    valid = True
                rec[sym] = float(val) if val is not None else None
        if valid:
            records.append(rec)
    df = pd.DataFrame(records).set_index("base_alloy")
    return df[list(_XLSX_ELEMENTS.values())].astype(float)


def parse_alloy_folder(folder_name: str) -> tuple[str, bool]:
    """'AZ31_extruded_heattreated' -> ('AZ31', True); 'Z1_extruded' -> ('Z1', False)."""
    heat_treated = folder_name.endswith("_heattreated")
    base = re.sub(r"_extruded(_heattreated)?$", "", folder_name)
    return base, heat_treated


def enumerate_conditions(database_root: str | Path) -> pd.DataFrame:
    """Walk database_24_06 and enumerate all alloy x (T, v) condition folders."""
    root = Path(database_root)
    records = []
    for alloy_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        base, ht = parse_alloy_folder(alloy_dir.name)
        for cond_dir in sorted(p for p in alloy_dir.iterdir() if p.is_dir()):
            m = _COND_RE.match(cond_dir.name)
            if m is None:
                continue
            records.append({
                "condition_id": f"{alloy_dir.name}_{cond_dir.name}",
                "alloy": alloy_dir.name,
                "base_alloy": base,
                "heat_treated": ht,
                "T_ext": float(m.group("T")),
                "v_ext": float(m.group("v")),
            })
    if not records:
        raise FileNotFoundError(f"No condition folders found under {root}")
    return pd.DataFrame(records)


def build_labels(database_root: str | Path, labels_xlsx: str | Path,
                 include_heat_treated: bool = False) -> pd.DataFrame:
    comp = read_composition_table(labels_xlsx)
    cond = enumerate_conditions(database_root)

    if not include_heat_treated:
        cond = cond[~cond["heat_treated"]].reset_index(drop=True)

    missing = sorted(set(cond["base_alloy"]) - set(comp.index))
    if missing:
        raise KeyError(
            f"Alloys in database without composition in xlsx: {missing}. "
            "Fix data_labels.xlsx or the name normalization map."
        )

    df = cond.join(comp, on="base_alloy")
    # Deterministic alloy class index over base alloys present (for aux classifier)
    classes = sorted(df["base_alloy"].unique())
    df["alloy_class_idx"] = df["base_alloy"].map({c: i for i, c in enumerate(classes)})

    cols = (["condition_id", "alloy", "base_alloy", "heat_treated",
             "alloy_class_idx"] + ELEMENTS + ["T_ext", "v_ext"])
    df = df[cols].sort_values("condition_id").reset_index(drop=True)

    # Sanity checks
    assert df["condition_id"].is_unique
    assert df[ELEMENTS].notna().all().all(), "NaN composition entries"
    assert df["T_ext"].between(150, 550).all()
    assert df["v_ext"].between(0.1, 10).all()
    return df


def main(config_path: str = "configs/paths.yaml") -> Path:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    df = build_labels(cfg["database_root"], cfg["labels_xlsx"],
                      cfg.get("include_heat_treated", False))
    out_dir = Path(cfg["data_root"]) / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "labels.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} conditions ({df['base_alloy'].nunique()} alloys) -> {out}")
    print(df.groupby("base_alloy").size().to_string())
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    args = ap.parse_args()
    main(args.config)
