"""
cluster_labeler.py  --- Generalizable cluster naming for the R-DPMM pipeline.
=============================================================================
Determines "which cluster corresponds to which category" for ANY dataset and
writes the result into the DOCX report. Two complementary, dataset-agnostic
mechanisms:

  (1) GROUND-TRUTH MAPPING  -- if the raw data carries one or more categorical
      columns (e.g. Product_Category, Warehouse) that were dropped before
      clustering, each cluster is mapped to its DOMINANT category by majority
      vote, with a PURITY score = % of the cluster belonging to that category.

  (2) DESCRIPTIVE PROFILE NAME -- always available, even with no labels at all.
      Each cluster is named from the z-scores of its feature means relative to
      the global mean (e.g. "High Order_Demand, Low Unit_Price, Short Lead").

Usage (standalone):
    python cluster_labeler.py \
        --original electronics_inventory.csv \
        --clustered electronics_inventory_clustered.csv \
        --docx-in  electronics_inventory_cluster_report.docx \
        --docx-out electronics_inventory_cluster_report_labeled.docx
"""

import argparse
import os
import re
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
# Substrings that disqualify a column from being a "category". NOTE: "name" is
# deliberately NOT here -- it would wrongly drop legit low-cardinality category
# columns like "Category Name"/"Department Name". True name fields (Customer
# Name, Product Name) are removed instead by the 2..MAX_CAT_CARDINALITY filter.
ID_HINTS = ("id", "code", "key", "index", "idx", "uuid", "serial", "date",
            "timestamp", "sku", "zipcode", "zip",
            # free-text / per-entity fields that are never a useful category:
            "image", "url", "http", "www", "email", "mail", "password",
            "phone", "street", "address", "description", "fname", "lname")
DEFAULT_CLUSTER_COL = "Cluster_Label"
DEFAULT_UNCERT_COL = "Uncertainty_Score"
Z_THRESHOLD = 0.60                                # |z| above which a feature is notable
MAX_CAT_CARDINALITY = 60     # max distinct levels for a "category". 60 keeps real
                             # taxonomies (DataCo Category Name ~51, Department ~11)
                             # but drops per-product identifiers (Product Name/Image
                             # ~118) that merely proxy product identity. Configurable.
CATEGORY_MAX_RATIO = 0.5     # also skip near-identifier cols (nunique/n above this)



def _read_table(path):
    """Robust reader: Excel by extension, else CSV with encoding fallback.
    Reuses robust_load.py if present; otherwise fully self-contained."""
    try:
        from robust_load import robust_load
        return robust_load(path)
    except Exception:
        pass
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".xls":
        return pd.read_excel(path, engine="xlrd")
    for e in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=e)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")   # latin-1 maps all 256 bytes


# ======================================================================
# 1. ALIGNMENT  --  join clustered labels back onto the original rows
# ======================================================================

def load_and_align(original_csv, clustered_csv,
                   cluster_col=DEFAULT_CLUSTER_COL,
                   uncert_col=DEFAULT_UNCERT_COL):
    """
    Returns (merged, feature_cols) where `merged` has every ORIGINAL column
    plus the cluster label, aligned row-for-row to the clustered output.

    Robust to the usual mismatch (the pipeline drops outlier rows, so the
    clustered CSV is shorter and has no index column): we inner-join on the
    shared numeric feature columns, which uniquely identify a row in practice.
    If an explicit '__orig_idx__' column exists, that is used instead.
    """
    orig = _read_table(original_csv)
    clu = _read_table(clustered_csv)
    if cluster_col not in clu.columns:
        raise ValueError(f"'{cluster_col}' not in clustered CSV columns {list(clu.columns)}")

    # feature columns = numeric columns shared by both files, minus label/uncertainty
    drop = {cluster_col, uncert_col}
    feature_cols = [c for c in clu.columns
                    if c not in drop and c in orig.columns
                    and pd.api.types.is_numeric_dtype(clu[c])
                    and pd.api.types.is_numeric_dtype(orig[c])]
    if not feature_cols:
        raise ValueError("No shared numeric feature columns to align on.")

    # Tier A: explicit preserved index
    if "__orig_idx__" in clu.columns:
        merged = orig.iloc[clu["__orig_idx__"].astype(int).values].copy().reset_index(drop=True)
        merged[cluster_col] = clu[cluster_col].values
        if uncert_col in clu.columns:
            merged[uncert_col] = clu[uncert_col].values
        return merged, feature_cols

    # Tier B: feature-value join (rounded keys)
    # Scope NaN-dropping to the join keys so sparse audit columns
    # (all-empty descriptions, mostly-empty zip codes) do not zero out the frame.
    orig = orig.dropna(subset=feature_cols).reset_index(drop=True)
    o = orig.copy()
    c = clu.copy()
    for k in feature_cols:
        o[k] = o[k].round(2)
        c[k] = c[k].round(2)
    keep = feature_cols + [cluster_col] + ([uncert_col] if uncert_col in c.columns else [])
    merged = (o.merge(c[keep], on=feature_cols, how="inner")
                .drop_duplicates(subset=feature_cols)
                .reset_index(drop=True))
    return merged, feature_cols


# ======================================================================
# 2. CANDIDATE CATEGORY COLUMNS  (ground-truth source, fully auto-detected)
# ======================================================================

def _candidate_reason(merged, col, skip):
    """Whether `col` can serve as a ground-truth category, with a reason string.
    Do NOT test `dtype == object`: pandas may load text as the PyArrow `string`
    dtype, which an object check would miss."""
    if col in skip:
        return False, "feature/label column"
    if any(h in col.lower() for h in ID_HINTS):
        return False, "name matches an ID hint"
    dt = merged[col]
    if pd.api.types.is_float_dtype(dt):
        return False, "continuous numeric"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return False, "datetime"
    if pd.api.types.is_bool_dtype(dt):
        return False, "boolean"
    n = len(merged)
    nun = int(dt.nunique(dropna=True))
    if nun < 2:
        return False, "constant"
    if nun > MAX_CAT_CARDINALITY:
        return False, f"too many levels ({nun} > {MAX_CAT_CARDINALITY})"
    if n and nun / n > CATEGORY_MAX_RATIO:
        return False, "near-unique (identifier-like)"
    return True, "ok"


def detect_category_columns(merged, feature_cols, cluster_col):
    """Usable category columns ranked by ASSOCIATION with the clustering
    (Adjusted Mutual Information), best first. AMI is chance-corrected, so an
    imbalanced column (e.g. mostly-one-country) cannot win on dominance alone."""
    skip = set(feature_cols) | {cluster_col, DEFAULT_UNCERT_COL}
    cands = [c for c in merged.columns if _candidate_reason(merged, c, skip)[0]]
    cands.sort(key=lambda c: _association(merged, cluster_col, c)["ami"], reverse=True)
    return cands


def diagnose_columns(merged, feature_cols, cluster_col):
    """Every non-feature column with its level count, whether it qualifies as a
    category (and why not), and its AMI with the clusters -- full transparency."""
    skip = set(feature_cols) | {cluster_col, DEFAULT_UNCERT_COL}
    rows = []
    for col in merged.columns:
        if col in skip:
            continue
        ok, reason = _candidate_reason(merged, col, skip)
        nun = int(merged[col].nunique(dropna=True))
        ami = (_association(merged, cluster_col, col)["ami"]
               if 2 <= nun <= 2000 else float("nan"))
        rows.append({"column": col, "levels": nun,
                     "candidate": ok, "reason": reason, "ami": ami})
    rows.sort(key=lambda r: (r["candidate"],
                             r["ami"] if r["ami"] == r["ami"] else -9), reverse=True)
    return rows


def _association(merged, cluster_col, col):
    """Chance-corrected association of the cluster partition with one column.
    Returns {'ami','cramers_v','mean_purity','n_levels'}."""
    from sklearn.metrics import adjusted_mutual_info_score
    sub = merged[[cluster_col, col]].dropna()
    out = {"ami": -1.0, "cramers_v": float("nan"),
           "mean_purity": float("nan"), "n_levels": int(sub[col].nunique())}
    if len(sub) < 2 or sub[col].nunique() < 2 or sub[cluster_col].nunique() < 2:
        return out
    a = sub[cluster_col].astype(str).to_numpy()
    b = sub[col].astype(str).to_numpy()
    out["ami"] = float(adjusted_mutual_info_score(a, b))
    ct = pd.crosstab(sub[cluster_col], sub[col])
    out["mean_purity"] = float((ct.max(axis=1) / ct.sum(axis=1)).mean())
    try:
        from scipy.stats import chi2_contingency
        chi2 = chi2_contingency(ct, correction=False)[0]
        nn = ct.to_numpy().sum(); r, k = ct.shape
        denom = nn * (min(r, k) - 1)
        out["cramers_v"] = float((chi2 / denom) ** 0.5) if denom > 0 else float("nan")
    except Exception:
        pass
    return out


def rank_categories(merged, feature_cols, cluster_col):
    """Candidate category columns with association scores, best-associated first."""
    return [{"column": c, **_association(merged, cluster_col, c)}
            for c in detect_category_columns(merged, feature_cols, cluster_col)]


# ======================================================================
# 3. GROUND-TRUTH MAPPING  (majority vote + purity)
# ======================================================================

def ground_truth_mapping(merged, cluster_col, cat_col):
    """For each cluster: dominant category, purity, and full distribution."""
    ct = pd.crosstab(merged[cluster_col], merged[cat_col])
    out = {}
    for cl in ct.index:
        row = ct.loc[cl]
        dom = row.idxmax()
        purity = float(row.max() / row.sum())
        out[cl] = {"category": str(dom),
                   "purity": purity,
                   "distribution": {str(k): int(v) for k, v in row.items() if v > 0}}
    return out


# ======================================================================
# 4. DESCRIPTIVE PROFILE NAMES  (works with NO labels)
# ======================================================================

def _friendly(feat):
    return feat.replace("_", " ").strip()

def descriptive_profiles(merged, feature_cols, cluster_col, z_thresh=Z_THRESHOLD):
    """
    Name each cluster from the z-scores of its feature means vs the global mean.
    Returns {cluster: {'name', 'tags': {feat: 'High'/'Low'/'Average'}, 'z': {...}}}.
    """
    g_mean = merged[feature_cols].mean()
    g_std = merged[feature_cols].std(ddof=0).replace(0, 1e-9)
    out = {}
    for cl, grp in merged.groupby(cluster_col):
        z = (grp[feature_cols].mean() - g_mean) / g_std
        tags, notable = {}, []
        for f in feature_cols:
            zf = float(z[f])
            if zf >= z_thresh:
                tags[f] = "High"
            elif zf <= -z_thresh:
                tags[f] = "Low"
            else:
                tags[f] = "Average"
            if abs(zf) >= z_thresh:
                notable.append((abs(zf), f, "High" if zf > 0 else "Low"))
        notable.sort(reverse=True)
        if notable:
            name = ", ".join(f"{lvl} {_friendly(f)}" for _, f, lvl in notable[:3])
        else:
            name = "Balanced / average across all features"
        out[cl] = {"name": name, "tags": tags,
                   "z": {f: round(float(z[f]), 2) for f in feature_cols}}
    return out


# ======================================================================
# 5. COMBINE
# ======================================================================

def build_labels(merged, feature_cols, cluster_col, category_col=None):
    cat_cols = detect_category_columns(merged, feature_cols, cluster_col)
    if category_col is not None and category_col in merged.columns:
        primary = category_col                       # explicit user override
        if category_col not in cat_cols:
            cat_cols = [category_col] + cat_cols
    else:
        primary = cat_cols[0] if cat_cols else None
    gt = ground_truth_mapping(merged, cluster_col, primary) if primary else {}
    desc = descriptive_profiles(merged, feature_cols, cluster_col)
    sizes = merged[cluster_col].value_counts().to_dict()

    labels = {}
    for cl in sorted(merged[cluster_col].unique()):
        entry = {"cluster": int(cl),
                 "n": int(sizes.get(cl, 0)),
                 "descriptive": desc[cl]["name"],
                 "tags": desc[cl]["tags"],
                 "z": desc[cl]["z"]}
        # A cluster can be absent from `gt` if ALL its rows have a missing value
        # in the chosen category column (it gets dropped from the cross-tab).
        # Such a cluster simply stays descriptive-only instead of crashing.
        if primary and cl in gt:
            entry["category_column"] = primary
            entry["category"] = gt[cl]["category"]
            entry["purity"] = gt[cl]["purity"]
            entry["distribution"] = gt[cl]["distribution"]
        labels[int(cl)] = entry
    return labels, primary, cat_cols


def label_string(entry, with_category=True):
    """Compact one-line label e.g. 'Premium-Server (100% pure) - High Unit Price, ...'."""
    if with_category and entry.get("category"):
        return f"{entry['category']} ({entry['purity']*100:.0f}% pure)"
    return entry["descriptive"]


# ======================================================================
# 6. DOCX AUGMENTATION
# ======================================================================

def _insert_paragraph_before(ref_para, text, doc, bold=False, size=None, style=None):
    from docx.shared import Pt
    new_p = ref_para.insert_paragraph_before(text, style=style)
    if new_p.runs:
        r = new_p.runs[0]
        r.bold = bold
        if size:
            r.font.size = Pt(size)
    return new_p


def _apply_grid_borders(table, header_fill="D5E8F0"):
    """Give a python-docx table visible grid borders + a shaded header row,
    independent of named styles (the report defines none)."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0"); el.set(qn("w:color"), "999999")
        borders.append(el)
    tblPr.append(borders)
    for cell in table.rows[0].cells:                 # shade header row
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), header_fill)
        cell._tc.get_or_add_tcPr().append(shd)


def augment_docx(docx_in, docx_out, labels, primary_col, cluster_col=DEFAULT_CLUSTER_COL):
    """
    1. Renames every 'Cluster N' header cell to 'Cluster N: <Category>'.
    2. Inserts a 'Cluster Identity & Naming' section (heading + table) right
       before the first numbered section of the report (falls back to append).
    """
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document(docx_in)

    cluster_re = re.compile(r"^\s*Cluster\s+(\d+)\s*$")

    # --- (1) rename cluster headers in every table cell ---
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                m = cluster_re.match(cell.text)
                if not m:
                    continue
                cl = int(m.group(1))
                if cl in labels and labels[cl].get("category"):
                    new_txt = f"Cluster {cl}: {labels[cl]['category']}"
                elif cl in labels:
                    short = labels[cl]["descriptive"].split(",")[0]
                    new_txt = f"Cluster {cl}: {short}"
                else:
                    continue
                # rewrite while keeping the first run's formatting
                if cell.paragraphs and cell.paragraphs[0].runs:
                    cell.paragraphs[0].runs[0].text = new_txt
                    for extra in cell.paragraphs[0].runs[1:]:
                        extra.text = ""
                else:
                    cell.text = new_txt

    # --- (2) locate insertion point: first heading beginning with a digit ---
    anchor = None
    for p in doc.paragraphs:
        if re.match(r"^\s*\d+[\.\)]?\s+\S", p.text):   # "1. Executive Summary"
            anchor = p
            break

    has_cat = primary_col is not None

    # Build the identity table at the end of the doc, then move it into place.
    heading_txt = "Cluster Identity & Naming"
    intro_txt = (
        f"Each cluster below is named two ways: a ground-truth category obtained by "
        f"majority vote over the '{primary_col}' column (with a purity score = share of "
        f"the cluster belonging to that category), and a descriptive profile derived from "
        f"how the cluster's feature means deviate from the global average."
        if has_cat else
        "No categorical ground-truth column was found, so clusters are named purely from "
        "their feature profiles (deviation of each cluster's feature means from the global "
        "average)."
    )

    cols = (["Cluster", "Category", "Purity", "Size", "Descriptive Profile"]
            if has_cat else ["Cluster", "Size", "Descriptive Profile"])
    table = doc.add_table(rows=1, cols=len(cols))
    for j, c in enumerate(cols):
        cell = table.rows[0].cells[j]
        cell.text = c
        if cell.paragraphs[0].runs:
            cell.paragraphs[0].runs[0].bold = True

    for cl in sorted(labels):
        e = labels[cl]
        row = table.add_row().cells
        if has_cat:
            row[0].text = f"Cluster {cl}"
            row[1].text = str(e.get("category", "-"))
            row[2].text = f"{e.get('purity', 0)*100:.1f}%"
            row[3].text = f"{e['n']:,}"
            row[4].text = e["descriptive"]
        else:
            row[0].text = f"Cluster {cl}"
            row[1].text = f"{e['n']:,}"
            row[2].text = e["descriptive"]

    _apply_grid_borders(table)

    # Move heading + intro + table before the anchor (or leave appended).
    if anchor is not None:
        # Order before the anchor: heading -> intro -> table.  insert_paragraph_before
        # always drops in immediately above the anchor, so insert heading first.
        h = _insert_paragraph_before(anchor, heading_txt, doc, bold=True, size=14)
        try:
            h.style = anchor.style            # match the report's section-heading look
            for r in h.runs:
                r.bold = True
        except Exception:
            pass
        _insert_paragraph_before(anchor, intro_txt, doc)
        anchor._p.addprevious(table._tbl)
    else:
        doc.add_heading(heading_txt, level=1)
        doc.add_paragraph(intro_txt)

    doc.save(docx_out)
    return docx_out



def labels_for_pipeline(original_csv, clustered_csv,
                        cluster_col=DEFAULT_CLUSTER_COL, category_col=None):
    """One-call helper for the R-DPMM pipeline. Returns a JSON-serializable dict
    {primary_category_column, candidate_columns, ranking, labels}. Pass
    category_col to force a specific ground-truth column."""
    merged, feats = load_and_align(original_csv, clustered_csv, cluster_col)
    labels, primary, cat_cols = build_labels(merged, feats, cluster_col, category_col)
    return {"primary_category_column": primary,
            "candidate_columns": cat_cols,
            "ranking": rank_categories(merged, feats, cluster_col),
            "labels": labels}

# ======================================================================
# 7. CLI
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Label R-DPMM clusters by category and profile.")
    ap.add_argument("--original", required=True)
    ap.add_argument("--clustered", required=True)
    ap.add_argument("--docx-in", default=None)
    ap.add_argument("--docx-out", default=None)
    ap.add_argument("--cluster-col", default=DEFAULT_CLUSTER_COL)
    ap.add_argument("--category-col", default=None,
                    help="Force this column as the ground-truth category.")
    args = ap.parse_args()

    merged, feats = load_and_align(args.original, args.clustered, args.cluster_col)
    ranking = rank_categories(merged, feats, args.cluster_col)
    labels, primary, cat_cols = build_labels(
        merged, feats, args.cluster_col, args.category_col)

    print(f"Aligned {len(merged):,} rows | features={feats}")
    print("Column scan (AMI = association with clusters; higher = better match):")
    print(f"    {'column':<26s} {'levels':>6s} {'AMI':>8s}  status")
    for r in diagnose_columns(merged, feats, args.cluster_col):
        ami = f"{r['ami']:+.3f}" if r['ami'] == r['ami'] else "    -   "
        flag = "CATEGORY" if r['candidate'] else f"skip ({r['reason']})"
        print(f"    {r['column']:<26s} {r['levels']:>6d} {ami:>8s}  {flag}")
    print(f"\nPrimary category column: {primary or 'NONE (descriptive only)'}\n")
    for cl in sorted(labels):
        e = labels[cl]
        cat = (f"{e['category']:<16} purity={e['purity']*100:5.1f}%"
               if e.get("category") else "(no category column)")
        print(f"  Cluster {cl}: {cat} | {e['descriptive']}  (n={e['n']:,})")

    if args.docx_in and args.docx_out:
        out = augment_docx(args.docx_in, args.docx_out, labels, primary, args.cluster_col)
        print(f"\nLabeled report written to: {out}")
    return labels


if __name__ == "__main__":
    main()
