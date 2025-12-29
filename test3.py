# diet2.py — robust Part 2 solver with friendlier path/engine handling
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import pulp as pl

# ---------- Defaults / auto-detection ----------
# Set this to your known file; used if no --excel and nothing in current folder matches diet*.xls*
DEFAULT_EXCEL_PATH = r"C:\Users\minyi\OneDrive\Desktop\ISYE 6501\Homework 11\Homework11_ISYE6501\diet.xls"

def auto_find_excel() -> str | None:
    cands = sorted(glob.glob("diet*.xls")) + sorted(glob.glob("diet*.xlsx"))
    return cands[0] if cands else None

def choose_engine_for(path: Path) -> str | None:
    # Prefer explicit engines for reliability
    if path.suffix.lower() == ".xlsx":
        return "openpyxl"     # pip install openpyxl
    else:
        return "xlrd"         # pip install xlrd2 (recommended on new Python)

# ---------- Protein detection ----------
PROTEIN_REGEX = re.compile(
    r"(beef|pork|chicken|turkey|lamb|veal|fish|salmon|tuna|sardine|cod|trout|shrimp|egg|ham|bacon|"
    r"sausage|duck|goose|anchovy|mackerel|tilapia|shellfish|clam|oyster|crab|lobster)",
    re.I,
)
def is_protein(name: str) -> bool:
    return bool(PROTEIN_REGEX.search(name or ""))

# ---------- Excel loader (robust to 1-sheet or 2-sheet layouts) ----------
def load_diet_excel(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    engine = choose_engine_for(path)

    # Try primary engine; if it fails on .xls, retry with xlrd2 (common on Py>=3.11)
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine=engine)
    except Exception as e:
        if path.suffix.lower() == ".xls":
            try:
                sheets = pd.read_excel(path, sheet_name=None, engine="xlrd2")
            except Exception:
                raise e
        else:
            raise e

    def sanitize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(how="all").dropna(how="all", axis=1)
        if df.empty or df.columns.size == 0:
            return pd.DataFrame()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    clean = [(nm, sanitize(df)) for nm, df in sheets.items()]
    clean = [(nm, df) for nm, df in clean if not df.empty and df.columns.size > 0]
    if not clean:
        raise ValueError("All sheets appear empty after cleaning.")

    def looks_like_foods(df: pd.DataFrame) -> bool:
        cols = [c.lower() for c in df.columns]
        name_ok  = any(("food" in c) or ("item" in c) for c in cols)
        price_ok = any("price" in c for c in cols)
        numeric_cols = df.select_dtypes(include="number").shape[1] >= 3
        return bool(name_ok and price_ok and numeric_cols)

    def looks_like_bounds(df: pd.DataFrame) -> bool:
        cols = [c.lower() for c in df.columns]
        has_min = any(c in ("min", "minimum") for c in cols)
        has_max = any(c in ("max", "maximum") for c in cols)
        return bool(has_min and has_max)

    foods_candidate, bounds_candidate = None, None
    for _, df in clean:
        if looks_like_foods(df):
            foods_candidate = df.copy()
        elif looks_like_bounds(df):
            bounds_candidate = df.copy()

    if foods_candidate is None:
        foods_candidate = max((df for _, df in clean), key=lambda d: d.shape[1]).copy()

    foods_df = foods_candidate.copy()
    name_col  = next((c for c in foods_df.columns if re.search(r"(food|item)", c, re.I)), None)
    price_col = next((c for c in foods_df.columns if re.search(r"price", c, re.I)), None)
    if not name_col or not price_col:
        raise ValueError(f"Could not find Food/Price columns. Columns: {list(foods_df.columns)}")

    # If bounds not separate, try block below foods
    if bounds_candidate is None:
        price_num = pd.to_numeric(foods_df[price_col], errors="coerce")
        if price_num.notna().any():
            last_food_idx = foods_df.index[price_num.notna()].max()
            after = foods_df.loc[last_food_idx + 1 :].copy()
            if not after.empty:
                keep = []
                for c in after.columns:
                    lc = str(c).lower()
                    if lc in ("min", "minimum", "max", "maximum") or lc in (name_col, "nutrient", "name"):
                        keep.append(c)
                tmp = after[keep].dropna(how="all") if keep else pd.DataFrame()
                if not tmp.empty:
                    ncol = None
                    for cand in (name_col, "nutrient", "name"):
                        if cand in tmp.columns:
                            ncol = cand
                            break
                    mincol = next((c for c in tmp.columns if str(c).lower() in ("min", "minimum")), None)
                    maxcol = next((c for c in tmp.columns if str(c).lower() in ("max", "maximum")), None)
                    if ncol and (mincol or maxcol):
                        b = tmp[[ncol] + [c for c in [mincol, maxcol] if c is not None and c in tmp.columns]].copy()
                        b = b.rename(columns={ncol: "nutrient"})
                        for c in b.columns:
                            if c != "nutrient":
                                b[c] = pd.to_numeric(b[c], errors="coerce")
                        if "minimum" in b.columns and "min" not in b.columns:
                            b["min"] = b["minimum"]
                        if "maximum" in b.columns and "max" not in b.columns:
                            b["max"] = b["maximum"]
                        if {"min", "max"} & set(b.columns):
                            # ✅ bugfix: use 'col' not 'c' in the list comprehension
                            bounds_candidate = b[["nutrient"] + [col for col in ["min", "max"] if col in b.columns]]
                        foods_df = foods_df.loc[:last_food_idx].copy()

    # Finalize foods_df
    nutrient_cols = [c for c in foods_df.columns if c not in (name_col, price_col)]
    num = foods_df[nutrient_cols].apply(pd.to_numeric, errors="coerce")
    foods_df = pd.concat([foods_df[[name_col, price_col]].reset_index(drop=True),
                          num.reset_index(drop=True)], axis=1)
    foods_df = foods_df.rename(columns={name_col: "Food", price_col: "Price"})
    foods_df["Food"] = foods_df["Food"].astype(str).str.strip()
    foods_df["Price"] = foods_df["Price"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    foods_df["Price"] = pd.to_numeric(foods_df["Price"], errors="coerce")
    foods_df = foods_df.dropna(subset=["Food", "Price"]).drop_duplicates(subset=["Food"])

    # Build bounds_df
    if bounds_candidate is None or bounds_candidate.empty:
        bounds_df = pd.DataFrame({"nutrient": num.columns, "min": 0.0, "max": np.inf}).set_index("nutrient")
    else:
        b = bounds_candidate.copy()
        b.columns = [str(c).strip() for c in b.columns]
        if "nutrient" not in b.columns:
            b = b.rename(columns={b.columns[0]: "nutrient"})
        b["nutrient"] = b["nutrient"].astype(str).str.strip()
        if "min" not in b.columns: b["min"] = 0.0
        if "max" not in b.columns: b["max"] = np.inf
        b["min"] = pd.to_numeric(b["min"], errors="coerce").fillna(0.0)
        b["max"] = pd.to_numeric(b["max"], errors="coerce").fillna(np.inf)
        bounds_df = b[["nutrient", "min", "max"]].drop_duplicates(subset=["nutrient"]).set_index("nutrient")

    nutrients = [c for c in foods_df.columns if c not in ("Food", "Price")]
    bounds_df = bounds_df.loc[bounds_df.index.intersection(nutrients)]
    foods_df[nutrients] = foods_df[nutrients].fillna(0.0)
    return foods_df, bounds_df

# ---------- Part 2 model + solve ----------
def solve_part2(
    foods_df: pd.DataFrame,
    bounds_df: pd.DataFrame,
    mode: str = "cost",
    cholesterol_col: str | None = None,
):
    foods = foods_df["Food"].tolist()

    # Objective coefficients
    if mode == "cost":
        coef = foods_df.set_index("Food")["Price"].to_dict()
    elif mode == "cholesterol":
        if cholesterol_col is None:
            for cand in ["Cholesterol", "cholesterol", "CHOLESTEROL (MG)", "Cholestrol"]:
                if cand in foods_df.columns:
                    cholesterol_col = cand
                    break
        if not cholesterol_col or cholesterol_col not in foods_df.columns:
            raise ValueError("Could not find a cholesterol column. Use --chol-col to specify.")
        coef = foods_df.set_index("Food")[cholesterol_col].to_dict()
    else:
        raise ValueError("mode must be 'cost' or 'cholesterol'")

    # Nutrients & amounts
    nutrients = bounds_df.index.tolist()
    amt: Dict[str, Dict[str, float]] = {n: foods_df.set_index("Food")[n].to_dict() for n in nutrients}

    # Protein set
    protein_set: Set[str] = {f for f in foods if is_protein(f)}

    # Model
    m = pl.LpProblem("Diet_Part2", sense=pl.LpMinimize)
    x = pl.LpVariable.dicts("serv", foods, lowBound=0, cat=pl.LpContinuous)
    y = pl.LpVariable.dicts("choose", foods, lowBound=0, upBound=1, cat=pl.LpBinary)

    # Objective
    m += pl.lpSum([float(coef[i]) * x[i] for i in foods])

    # x_i >= 0.1 * y_i
    for i in foods:
        m += x[i] >= 0.1 * y[i], f"min_serv_if_chosen_{i}"

    # Nutrient constraints
    for n in nutrients:
        m += pl.lpSum([float(amt[n][i]) * x[i] for i in foods]) >= float(bounds_df.loc[n, "min"]), f"min_{n}"
        if np.isfinite(bounds_df.loc[n, "max"]):
            m += pl.lpSum([float(amt[n][i]) * x[i] for i in foods]) <= float(bounds_df.loc[n, "max"]), f"max_{n}"

    # At most one of celery / frozen broccoli
    celery_names = [i for i in foods if re.search(r"celery", i, re.I)]
    broc_names   = [i for i in foods if re.search(r"broccoli", i, re.I)]
    if celery_names or broc_names:
        m += pl.lpSum([y[i] for i in celery_names + broc_names]) <= 1, "at_most_one_celery_broccoli"

    # At least 3 protein items
    if protein_set:
        m += pl.lpSum([y[i] for i in protein_set]) >= 3, "at_least_3_proteins"

    # Solve
    m.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[m.status]
    objective_value = m.objective.value()

    servings = {i: x[i].value() for i in foods if (x[i].value() or 0.0) > 1e-7}
    chosen   = {i: int(round(y[i].value() or 0.0)) for i in foods if (y[i].value() or 0.0) > 0.5}
    protein_count = len(set(chosen.keys()) & protein_set) if protein_set else 0
    celery_broc_ok = (sum(1 for f in chosen if (f in celery_names) or (f in broc_names)) <= 1)

    return {
        "status": status,
        "objective": objective_value,
        "servings": servings,
        "chosen": chosen,
        "protein_count": protein_count,
        "celery_broc_ok": celery_broc_ok,
        "mode": mode,
    }

# ---------- CLI ----------
def resolve_excel_path(cli_arg: str | None) -> str:
    # 1) CLI arg
    if cli_arg:
        return cli_arg
    # 2) auto-detect in current folder
    found = auto_find_excel()
    if found:
        return found
    # 3) default known path
    if Path(DEFAULT_EXCEL_PATH).exists():
        return DEFAULT_EXCEL_PATH
    raise SystemExit(
        "No Excel file given, none matched diet*.xls[x] in this folder, "
        "and DEFAULT_EXCEL_PATH does not exist.\n"
        'Pass --excel "C:\\path\\to\\diet.xls"'
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", "-e", default=None, help="Path to diet.xls or diet_large.xls")
    ap.add_argument("--mode", choices=["cost", "cholesterol"], default="cost",
                    help="Objective: minimize 'cost' or 'cholesterol'")
    ap.add_argument("--chol-col", default=None, help="Cholesterol column name in diet_large (if nonstandard)")
    args = ap.parse_args()

    excel_path = resolve_excel_path(args.excel)
    foods_df, bounds_df = load_diet_excel(excel_path)
    res = solve_part2(foods_df, bounds_df, mode=args.mode, cholesterol_col=args.chol_col)

    print("\n== Part 2 Result ==")
    print("File:", excel_path)
    print("Mode:", res["mode"])
    print("Status:", res["status"])
    if res["mode"] == "cost":
        print("Minimum daily cost: ${:.4f}".format(res["objective"] if res["objective"] is not None else float("nan")))
    else:
        print("Minimum daily cholesterol: {:.4f}".format(res["objective"] if res["objective"] is not None else float("nan")))
    print("Protein items chosen (count):", res["protein_count"])
    print("Celery/Broccoli constraint satisfied?:", "Yes" if res["celery_broc_ok"] else "No")

    if res["chosen"]:
        print("\nChosen foods and servings:")
        for f in sorted(res["chosen"].keys(), key=str.lower):
            print(f"  {f}: selected=1, servings={res['servings'].get(f, 0.0):.4f}")
    else:
        print("\n(No items selected?) Check bounds/feasibility.")

if __name__ == "__main__":
    main()
