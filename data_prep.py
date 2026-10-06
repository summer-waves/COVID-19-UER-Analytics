from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CORE_FILE = DATA_DIR / "COVID-19 Project COVID Cases.xlsx" 
OWID_FILE = DATA_DIR / "owid-covid-data.csv"      
STIMULUS_FILE = DATA_DIR / "stimulus.csv"        

CONTEXT_COLS = ["stringency_index", "people_fully_vaccinated_per_hundred", "stimulus_pct_gdp"]


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    as_text = pd.to_datetime(s.astype(str).str.strip(), format="%m/%y", errors="coerce")
    missing = as_text.isna()
    if missing.any():
        as_text[missing] = pd.to_datetime(s[missing], errors="coerce")
    return as_text


def load_core(path=CORE_FILE) -> pd.DataFrame:
    df = pd.read_excel(path) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["date"] = _parse_dates(df["date"]).dt.to_period("M").dt.to_timestamp()
    df["new_covid_cases"] = pd.to_numeric(df["new_covid_cases"], errors="coerce")
    df["unemployment_rate"] = pd.to_numeric(df["unemployment_rate"], errors="coerce")
    df = df.dropna(subset=["date", "country"]).sort_values(["country", "date"])
    return df.reset_index(drop=True)


def _load_owid(path=OWID_FILE) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    header = pd.read_csv(path, nrows=0).columns
    name_col = "country" if "country" in header else "location"
    wanted = [name_col, "date", "stringency_index", "people_fully_vaccinated_per_hundred"]
    present = [c for c in wanted if c in header]
    missing = [c for c in wanted if c not in header]
    if missing:
        print(f"[OWID] columns not found, skipping: {missing}")
    agg = {c: ("mean" if c == "stringency_index" else "max")
           for c in present if c not in (name_col, "date")}
    if not agg:
        return None
    o = pd.read_csv(path, usecols=present, parse_dates=["date"])
    o["date"] = o["date"].dt.to_period("M").dt.to_timestamp()
    return (o.groupby([name_col, "date"]).agg(agg).reset_index()
              .rename(columns={name_col: "country"}))


def _load_stimulus(path=STIMULUS_FILE) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    s = pd.read_csv(path)
    s["date"] = _parse_dates(s["date"]).dt.to_period("M").dt.to_timestamp()
    return s[["country", "date", "stimulus_pct_gdp"]]


def load_all() -> pd.DataFrame:
    df = load_core()
    for extra in (_load_owid(), _load_stimulus()):
        if extra is not None:
            df = df.merge(extra, on=["country", "date"], how="left")
    # Vaccination is cumulative and unreported early on -> 0 before the first report;
    # other context columns are carried forward within each country.
    if "people_fully_vaccinated_per_hundred" in df:
        df["people_fully_vaccinated_per_hundred"] = df["people_fully_vaccinated_per_hundred"].fillna(0)
    for c in CONTEXT_COLS:
        if c in df:
            df[c] = df.groupby("country")[c].transform(lambda s: s.ffill().bfill())
    return df


def available_context(df: pd.DataFrame) -> list[str]:
    return [c for c in CONTEXT_COLS if c in df.columns and df[c].notna().any()]
