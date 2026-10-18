"""Download CUAD v1 (Contract Understanding Atticus Dataset, CC BY 4.0) into data/raw/.

Source: Zenodo record 4595826 (The Atticus Project; Hendrycks et al., 2021).
Only CUAD_v1.json (SQuAD-style: full contract text + lawyer-labelled clause spans)
is extracted; the PDFs are not needed because the text version is the same contract.

Usage:  python scripts/00_download.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "CUAD_v1.json"
    if out.exists():
        print(f"already there: {out.relative_to(ROOT)}")
        return
    print(f"downloading {URL} (~106 MB)")
    with urllib.request.urlopen(URL) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in z.namelist() if n.endswith("CUAD_v1.json"))
    out.write_bytes(z.read(name))
    for n in z.namelist():                       # keep the category definitions too
        if n.endswith("category_descriptions.csv"):
            (RAW / "category_descriptions.csv").write_bytes(z.read(n))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
