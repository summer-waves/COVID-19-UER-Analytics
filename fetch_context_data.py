"""Downloads Our World in Data's COVID dataset (vaccination + stringency) into data/.
Run once:  python fetch_context_data.py
If the URL ever stops working, download 'owid-covid-data.csv' from the COVID-19 page on
ourworldindata.org and save it as data/owid-covid-data.csv.
"""
import urllib.request
from pathlib import Path

URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
dest = Path(__file__).parent / "data" / "owid-covid-data.csv"
dest.parent.mkdir(exist_ok=True)
print(f"Downloading {URL} ...")
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
    f.write(resp.read())
print(f"Saved to {dest}")
