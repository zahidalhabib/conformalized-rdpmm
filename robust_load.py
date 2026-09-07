"""
robust_load.py --- one loader for (almost) any tabular dataset.

Handles:
  * Excel       .xlsx / .xlsm / .xls   -> pandas Excel engines
  * Text/CSV    any encoding (UTF-8, UTF-8-BOM, Windows-1252, Latin-1, ...)
                with automatic detection + an ordered fallback chain
  * Delimiter   optional auto-sniff (comma / semicolon / tab / pipe)

Drop-in replacement for `pd.read_csv(path)` in the R-DPMM pipeline.
Dependencies: pandas; openpyxl for .xlsx (pip install openpyxl); xlrd for .xls.
"""

import os
import pandas as pd


def robust_load(path, sniff_delimiter=True, **kwargs):
    ext = os.path.splitext(str(path))[1].lower()

    # ---------- Excel ----------
    if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return pd.read_excel(path, engine="openpyxl", **kwargs)
    if ext == ".xls":
        return pd.read_excel(path, engine="xlrd", **kwargs)       # legacy binary
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)

    # ---------- CSV / text ----------
    # 1) try to sniff the encoding (optional libs; both are best-effort)
    enc = None
    try:
        from charset_normalizer import from_path
        best = from_path(path).best()
        enc = best.encoding if best else None
    except Exception:
        try:
            import chardet
            with open(path, "rb") as fh:
                enc = chardet.detect(fh.read(200_000)).get("encoding")
        except Exception:
            enc = None

    # 2) optional delimiter sniff on a decodable sample
    if sniff_delimiter and "sep" not in kwargs and "delimiter" not in kwargs:
        import csv
        for probe in (enc, "utf-8", "cp1252", "latin-1"):
            if not probe:
                continue
            try:
                with open(path, "r", encoding=probe, errors="strict") as fh:
                    sample = fh.read(8192)
                kwargs["sep"] = csv.Sniffer().sniff(
                    sample, delimiters=",;\t|").delimiter
                break
            except Exception:
                continue

    # 3) ordered encoding fallbacks; latin-1 maps all 256 bytes -> never raises
    for e in [enc, "utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        if not e:
            continue
        try:
            return pd.read_csv(path, encoding=e, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1", **kwargs)         # guaranteed


if __name__ == "__main__":
    import sys
    df = robust_load(sys.argv[1])
    print(f"loaded {len(df):,} rows x {len(df.columns)} cols")
    print("columns:", list(df.columns)[:12])
