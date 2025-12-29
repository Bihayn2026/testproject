# -*- coding: utf-8 -*-
"""
Diet problem (Part 1): Minimize cost subject to nutrient min/max constraints.
Works with diet.xls or diet.xlsx. Tested with pandas>=1.5, PuLP>=2.7.

Run:
    python diet_part1.py [optional_path_to_excel]
"""

from __future__ import annotations
import sys, re, math
from pathlib import Path
import pandas as pd
import numpy as np
import pulp as pl

# ------------ Loader ------------
def load_diet_excel(path: str | Path):
    """
    Returns:
        foods_df: columns -> ['Food','Price', <nutrients...>] (numeric nutrients)
        bounds_df: index = nutrient names, columns -> ['min','max'] (floats; 'max' may be +inf)
    Handles:
      - Two-sheet files: foods + bounds
      - One-sheet files: bounds "block" at the bottom
    """
    path = Path(path)
    # Pick engine depending on extension; xlsx=openpyxl, xls=xlrd/xlrd2
    engine = None
    if path.suffix.lower() == ".xlsx":
        engine = "openpyxl"

    xls = pd.read_excel("C:\Users\minyi\OneDrive\Desktop\ISYE 6501\Homework 11\Homework11_ISYE6501\diet.xls", sheet1=None, engine=engine)

    # Heuristics to identify foods vs bounds
    def looks_like_foods(df):
        cols = [str(c).strip().lower() for c in df.columns]
        name_ok  = any(re.search(r"(food|item)", c) for c in cols)
        price_ok = any(re.search(r"price", c) for c in cols)
        numeric_cols = df.select_dtypes("number").shape[1] >= 3
        return name_ok and price_ok and numeric_cols

    def looks_like_bounds(df):
        cols = [str(c).strip().lower() for c in df.columns]
        has_min = any(c in ("min", "minimum") for c in cols)
        has_max = any(c in ("max", "maximum") for c in cols)
        return has_min and has_max

    foods_candidate, bounds_candidate = None, None
    for _, df in xls.items():
        if looks_like_foods(df):
            foods_candidate = df.copy()
        elif looks_like_bounds(df):
            bounds_candidate = df.copy()

    if foods_candidate is None:
        # Fallback to widest sheet
        foods_candidate = max(xls.values(), key=lambda d: d.shape[1]).copy()

    foods_df = foods_candidate.copy()
    foods_df = foods_df.dropna(how="all")
    # Normalize col names
    foods_df = foods_df.rename(columns={c: str(c).strip() for c in foods_df.columns})
    name_col  = next((c for c in foods_df.columns if re.search(r"(food|item)", c, re.I)), None)
    price_col = next((c for c in foods_df.columns if re.search(r"price", c, re.I)), None)
    if not name_col or not price_col:
        raise ValueError("Could not find 'Food'/'Price' columns in the foods sheet.")

    # If no explicit bounds sheet, try to split a min/max block below the foods table
    if bounds_candidate is None:
        mask_price = pd.to_numeric(foods_df[price_col], errors="coerce").notna()
        if mask_price.any():
            last_food_idx = foods_df.index[mask_price].max()
            maybe_bounds = foods_df.loc[last_food_idx+1:].copy()
            if not maybe_bounds.empty:
                keep = []
                for c in maybe_bounds.columns:
                    lc = str(c).lower()
                    if lc in ("min","minimum","max","maximum") or lc in (name_col,"nutrient","name"):
                        keep.append(c)
                if keep:
                    tmp = maybe_bounds[keep].dropna(how="all")
                    ncol = None
                    for cand in (name_col,"nutrient","name"):
                        if cand in tmp.columns: ncol = cand; break
                    mincol = next((c for c in tmp.columns if str(c).lower() in ("min","minimum")), None)
                    maxcol = next((c for c in tmp.columns if str(c).lower() in ("max","maximum")), None)
                    if ncol and (mincol or maxcol):
                        b = tmp[[ncol] + [c for c in [mincol,maxcol] if c]].copy()
                        b = b.rename(columns={ncol: "nutrient"})
                        for c in b.columns:
                            if c != "nutrient":
                                b[c] = pd.to_numeric(b[c], errors="coerce")
                        if "minimum" in b.columns and "min" not in b.columns: b["min"] = b["minimum"]
                        if "maximum" in b.columns and "max" not in b.columns: b["max"] = b["maximum"]
                        if "min" in b.columns or "max" in b.columns:
                            bounds_candidate = b[["nutrient"] + [col for col in ["min","max"] if col in b.columns]].dropna(how="all", subset=["min","max"])
                        foods_df = foods_df.loc[:last_food_idx].copy()

    # Keep only numeric nutrients (+ Food, Price)
    nutrient_cols = [c for c in foods_df.columns if c not in (name_col, price_col)]
    num_nutrients = foods_df[nutrient_cols].apply(pd.to_numeric, errors="coerce")
    foods_df = pd.concat([foods_df[[name_col, price_col]].reset_index(drop=True),
                          num_nutrients.reset_index(drop=True)], axis=1)
    foods_df = foods_df.rename(columns={name_col:"Food", price_col:"Price"})
    foods_df["Food"]  = foods_df["Food"].astype(str).str.strip()
    # clean price to numeric
    foods_df["Price"] = (foods_df["Price"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True))
    foods_df["Price"] = pd.to_numeric(foods_df["Price"], errors="coerce")

    # Build bounds
    if bounds_candidate is None:
        # default: 0 <= nutrient <= +inf (you can tighten later if your file requires)
        bounds_df = pd.DataFrame({"nutrient": num_nutrients.columns, "min": 0.0, "max": np.inf}).set_index("nutrient")
    else:
        b = bounds_candidate.copy()
        b["nutrient"] = b["nutrient"].astype(str).str.strip()
        if "min" not in b.columns: b["min"] = 0.0
        if "max" not in b.columns: b["max"] = np.inf
        b["min"] = pd.to_numeric(b["min"], errors="coerce").fillna(0.0)
        b["max"] = pd.to_numeric(b["max"], errors="coerce")
        b["max"] = b["max"].fillna(np.inf)
        bounds_df = b[["nutrient","min","max"]].drop_duplicates(subset=["nutrient"]).set_index("nutrient")

    # Keep only bounds for nutrients we actually have
    nutrients = [c for c in foods_df.columns if c not in ("Food","Price")]
    bounds_df = bounds_df.loc[bounds_df.index.intersection(nutrients)]

    # Final cleaning
    foods_df = foods_df.dropna(subset=["Food","Price"])
    foods_df = foods_df.drop_duplicates(subset=["Food"], keep="first")
    foods_df[nutrients] = foods_df[nutrients].fillna(0.0)

    return foods_df, bounds_df


# ------------ Model (Part 1) ------------
def solve_min_cost(foods_df: pd.DataFrame, bounds_df: pd.DataFrame):
    foods = foods_df["Food"].tolist()
    price = foods_df.set_index("Food")["Price"].to_dict()
    # ensure no stray items without price
    foods = [i for i in foods if i in price]

    nutrients = bounds_df.index.tolist()
    amt = {n: foods_df.set_index("Food")[n].to_dict() for n in nutrients}

    m = pl.LpProblem("Diet_MinCost", sense=pl.LpMinimize)
    x = pl.LpVariable.dicts("serv", foods, lowBound=0, cat=pl.LpContinuous)

    # objective (use list to avoid generator quirks)
    m += pl.lpSum([float(price[i]) * x[i] for i in foods])

    # constraints
    for n in nutrients:
        # min
        m += pl.lpSum([float(amt[n][i]) * x[i] for i in foods]) >= float(bounds_df.loc[n, "min"]), f"min_{n}"
        # max (only if finite)
        if np.isfinite(bounds_df.loc[n, "max"]):
            m += pl.lpSum([float(amt[n][i]) * x[i] for i in foods]) <= float(bounds_df.loc[n, "max"]), f"max_{n}"

    m.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[m.status]
    cost = m.objective.value()
    sol = {i: x[i].value() for i in foods if (x[i].value() or 0) > 1e-7}
    return status, cost, sol


# ------------ CLI ------------
def main():
    # Path handling: raw string recommended on Windows
    path = sys.argv[1] if len(sys.argv) > 1 else "diet.xls"
    foods_df, bounds_df = load_diet_excel(path)

    # Quick peek
    print("\n== Loaded file ==")
    print("Foods rows:", len(foods_df), " Nutrients:", len([c for c in foods_df.columns if c not in ('Food','Price')]))
    print("Bounds rows:", len(bounds_df))
    print("\nFoods preview:")
    print(foods_df[['Food','Price']].head())

    # Solve
    status, cost, sol = solve_min_cost(foods_df, bounds_df)

    print("\n== Part 1: Minimize Cost ==")
    print("Status:", status)
    print("Minimum daily cost: ${:.4f}".format(cost if cost is not None and not math.isnan(cost) else float("nan")))
    if sol:
        print("\nSelected foods (servings):")
        for k in sorted(sol.keys(), key=str.lower):
            print(f"  {k}: {sol[k]:.4f}")
    else:
        print("No positive servings found. Check constraints/bounds.")

if __name__ == "__main__":
    main()
