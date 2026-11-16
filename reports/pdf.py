"""PDF report builder and the small summary tables the model can put in it.

These three functions are loaded into the namespace of the model's code.
The table is drawn cell by cell because the installed fpdf version has no FontFace/table API.
"""
import os
from io import BytesIO

import polars as pl
from fpdf import FPDF

from config import PDF_DIR


def _clean(t) -> str:
    """FPDF's core fonts are Latin-1 only, so replace anything else."""
    return str(t).encode("latin-1", "replace").decode("latin-1")


def _r(x):
    """Round to 2 decimals, keeping None for empty columns."""
    return None if x is None else round(float(x), 2)


def numeric_summary(df: pl.DataFrame) -> pl.DataFrame:
    """One row per numeric column: a transposed, readable describe()."""
    cols = [c for c, t in df.schema.items() if t.is_numeric()]
    return pl.DataFrame({
        "column": cols,
        "mean":   [_r(df[c].mean()) for c in cols],
        "std":    [_r(df[c].std()) for c in cols],
        "min":    [_r(df[c].min()) for c in cols],
        "median": [_r(df[c].median()) for c in cols],
        "max":    [_r(df[c].max()) for c in cols],
        "nulls":  [df[c].null_count() for c in cols],
    })


def category_summary(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Count and share of each value in a text column."""
    return (df.group_by(col).len().sort("len", descending=True)
              .with_columns((pl.col("len") / df.height * 100).round(1).alias("percent"))
              .rename({"len": "count"}))


def _add_table(pdf: FPDF, df: pl.DataFrame) -> None:
    w = pdf.epw / len(df.columns)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 241)
    for c in df.columns:
        pdf.cell(w, 7, _clean(c), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for row in df.rows():
        for v in row:
            txt = f"{v:,.2f}" if isinstance(v, float) else ("" if v is None else v)
            pdf.cell(w, 6, _clean(txt), border=1)
        pdf.ln()


def make_pdf_report(filename: str, title: str, sections: list) -> str:
    """sections: list of dicts with 'heading' plus one of
    'text' (str), 'bullets' (list of str), 'table' (pl.DataFrame) or 'fig' (Plotly figure)."""
    pdf = FPDF(orientation="L")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    for s in sections:
        s = {**s, **s.get("content", {})}  # tolerate {"heading": ..., "content": {...}}
        if s.get("heading"):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, _clean(s["heading"]), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        if "text" in s:
            pdf.multi_cell(0, 6, _clean(s["text"]), new_x="LMARGIN", new_y="NEXT")
        for b in s.get("bullets", []):
            pdf.multi_cell(0, 6, _clean("- " + str(b)), new_x="LMARGIN", new_y="NEXT")
        if "table" in s:
            _add_table(pdf, s["table"])
        if "fig" in s:
            png = s["fig"].to_image(format="png", width=1100, height=500)  # needs kaleido
            pdf.image(BytesIO(png), w=pdf.epw)
        pdf.ln(4)

    path = os.path.join(PDF_DIR, filename)
    pdf.output(path)
    return path
