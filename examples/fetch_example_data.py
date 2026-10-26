"""
Download the example catchment used in the README.

The data are the "L0123001" example catchment (360 km², daily 1984-2012:
precipitation, PET, air temperature and discharge) shipped with the airGR
R package (Coron et al., 2017; INRAE). They are downloaded from the CRAN
mirror on GitHub and converted to CSV - they are not redistributed here.

    pip install pyreadr
    python examples/fetch_example_data.py
"""

from pathlib import Path
from urllib.request import urlretrieve
import tempfile

import pyreadr

URL = "https://raw.githubusercontent.com/cran/airGR/master/data/L0123001.rda"
OUT = Path(__file__).parent / "data" / "L0123001.csv"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        rda = Path(tmp) / "L0123001.rda"
        urlretrieve(URL, rda)
        obs = pyreadr.read_r(str(rda))["BasinObs"]

    df = obs.rename(columns={"DatesR": "Date", "E": "PET"})
    df["Q"] = df["Qls"] / 1000.0  # l/s -> m3/s
    df = df[["Date", "P", "PET", "T", "Q"]]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} days to {OUT}")


if __name__ == "__main__":
    main()
