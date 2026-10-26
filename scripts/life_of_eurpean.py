"""Load Eurostat time-use data and fit per-country activity models.

Produces:
    models[country]['percentage'][activity]  -> object with .generate(n)  (participation rate, %)
    models[country]['time'][activity]        -> object with .generate(n)  (minutes per day)
"""
import numpy as np
import pandas as pd
from distfit import distfit
from tqdm import tqdm
import pypickle
import os
import hashlib

INPUT_FILE = "estat_tus_20age.csv"
OUTPUT_FILE = "eurostat_time_use_labeled.csv"

# Durations must be conditional on participation (PTP_TIME) because they are
# paired with participation rates (PTP_RT) in the simulation. Using TIME_SP
# (population average, including non-participants) here would apply the
# participation rate twice.
DURATION_UNIT = "TIME_SP" # TIME_SP or PTP_TIME
RATE_UNIT = "PTP_RT"

# Only keep leaf categories that map onto the simulation's states.
# AC0 (all personal care), AC02 (eating and drinking) and AC022 (drinking)
# are parents/overlaps of these and would double count if kept.
# Everything else (household, leisure, ...) is covered by "Other".
ACTIVITY_MAPPING = {
    "AC01": "Sleeping",
    "AC021": "Eating",
    "AC03": "Personal care",
    "AC1": "Employment",
    "AC2": "Study",
    "AC4": "Volunteer work",
    "AC9": "Travel",
    # Evening / leisure activities (scheduled after the evening trip).
    "AC3": "Household",
    "AC5": "Socialising",
    "AC6": "Sports",
    "AC7": "Hobbies",
    "AC8": "Media",
}

COUNTRY_MAPPING = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
    "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IS": "Iceland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg",
    "LV": "Latvia", "MT": "Malta", "NL": "Netherlands", "NO": "Norway",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "RS": "Serbia",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "UK": "United Kingdom",
    "EU27_2020": "European Union",
}

# With only a handful of observations per country/activity (e.g. one per sex),
# distfit's "popular" search is meaningless and can crash. Below this many
# points a simple normal model is used instead.
MIN_POINTS_FOR_FIT = 5


def _hhmm_to_minutes(s):
    if pd.isna(s):
        return np.nan
    hours, minutes = str(s).split(":")
    return int(hours) * 60 + int(minutes)


def load_time_use(path=INPUT_FILE, sexes=("M", "F"), ages=("TOTAL",)):
    """Return a tidy long DataFrame: country, sex, activity, unit, time_min, percentage.

    `sexes` avoids mixing the total ("T") with its own male/female components.
    `ages` keeps only these age classes (default the overall "TOTAL"). Without
    this, every age class in the file is averaged together, which badly distorts
    rates such as Study. Pass None to keep every category.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    geo_col = next((c for c in df.columns if c.lower().startswith("geo")), None)
    if geo_col is None:
        raise KeyError("No 'geo\\TIME_PERIOD' column found in the input file.")

    # Last column looks like "AT\t12:30 u": country code, tab, value (+ flag).
    parts = df[geo_col].astype(str).str.split("\t", n=1, expand=True)
    if parts.shape[1] < 2:
        raise ValueError("Expected a tab-separated country/value column.")
    df["geo"] = parts[0].str.strip()

    # Drop Eurostat flags (' u', ' e', ' b', ...) and treat ':' as missing.
    value = (
        parts[1].str.strip()
        .str.replace(r"\s+[A-Za-z]+$", "", regex=True)
        .str.strip()
        .replace({":": np.nan, "": np.nan})
    )

    out = pd.DataFrame({
        "country": df["geo"].map(COUNTRY_MAPPING).fillna(df["geo"]),
        "sex": df["sex"].astype(str).str.strip() if "sex" in df.columns else "T",
        "age": df["age"].astype(str).str.strip() if "age" in df.columns else "TOTAL",
        "activity": df["acl18"].astype(str).str.strip().map(ACTIVITY_MAPPING),
        "unit": df["unit"].astype(str).str.strip(),
        "value": value,
    })

    out = out[out["activity"].notna() & out["value"].notna()]
    out = out[out["unit"].isin([RATE_UNIT, DURATION_UNIT])].copy()
    if sexes is not None:
        out = out[out["sex"].isin(sexes)].copy()
    if ages is not None:
        if not out["age"].isin(ages).any():
            raise ValueError(f"None of the ages {ages} found; file has {sorted(out['age'].unique())}")
        out = out[out["age"].isin(ages)].copy()

    out["time_min"] = np.nan
    out["percentage"] = np.nan

    is_time = out["unit"] == DURATION_UNIT
    is_rate = out["unit"] == RATE_UNIT
    out.loc[is_time, "time_min"] = out.loc[is_time, "value"].map(_hhmm_to_minutes)
    out.loc[is_rate, "percentage"] = pd.to_numeric(
        out.loc[is_rate, "value"], errors="coerce"
    )

    out = out.drop(columns="value")
    # Drop rows whose value could not be parsed.
    out = out[out["time_min"].notna() | out["percentage"].notna()]
    return out.reset_index(drop=True)


class NormalFallback:
    """Minimal stand-in for a fitted distribution when data are too sparse."""

    def __init__(self, values, lo=0.0, hi=None):
        self.mean = float(np.mean(values))
        self.std = max(float(np.std(values)), 0.05 * abs(self.mean), 1e-6)
        self.lo, self.hi = lo, hi

    def generate(self, n):
        return np.clip(np.random.normal(self.mean, self.std, size=n), self.lo, self.hi)


def _fit(values, lo, hi):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    if len(values) >= MIN_POINTS_FOR_FIT and np.ptp(values) > 0:
        model = distfit(distr="popular", verbose=None, bins=50)
        model.fit_transform(values)
        # model.plot()
        return model
    return NormalFallback(values, lo, hi)


def fit_models(df):
    """Build models[country]['time'|'percentage'][activity]."""

    models = {}
    for country in tqdm(df["country"].unique()):
        sub = df[df["country"] == country]
        models[country] = {"time": {}, "percentage": {}}
        for activity in sub["activity"].unique():
            rows = sub[sub["activity"] == activity]
            t = _fit(rows.loc[rows["unit"] == DURATION_UNIT, "time_min"], 0.0, 24 * 60)
            p = _fit(rows.loc[rows["unit"] == RATE_UNIT, "percentage"], 0.0, 100.0)
            # Skip activities that lack either model, but say so.
            if t is not None and p is not None:
                models[country]["time"][activity] = t
                models[country]["percentage"][activity] = p
            else:
                print(f"WARNING {country}: {activity} skipped "
                      f"(time data: {t is not None}, participation data: {p is not None})")

    return models


if __name__ == "__main__":
    data = load_time_use()
    data.to_csv(OUTPUT_FILE, index=False)
    print("Rows:", len(data))
    print("Countries:", data["country"].nunique())
    print("Activities:", data["activity"].nunique())
    print(data["sex"].value_counts())
    print(data["activity"].value_counts())

    models = fit_models(data)

    print(sorted(models))
    print(models["Netherlands"]["time"]["Sleeping"].generate(5))
