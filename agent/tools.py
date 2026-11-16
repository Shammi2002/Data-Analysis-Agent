"""The `complete_python_task` tool: runs the model's Python code on the loaded datasets."""
import os
import pickle
import uuid
from contextlib import redirect_stdout
from io import StringIO

import polars as pl
import pypdf
import streamlit as st
from langchain_core.tools import tool

from config import IMAGE_DIR, MIN_PDF_BYTES, PDF_DIR
from reports.pdf import category_summary, make_pdf_report, numeric_summary

SAFE_TYPES = (pl.DataFrame, int, float, str, list, dict, bool)  # kinds of variables kept between calls


def build_namespace(dataset_names: list) -> dict:
    """Everything the model's code can see: `pl`, report helpers, variables saved earlier and the datasets."""
    namespace = {
        "pl": pl,
        "numeric_summary": numeric_summary,
        "category_summary": category_summary,
        "make_pdf_report": make_pdf_report,
        **st.session_state.get("_persistent_vars", {}),
    }
    loaded = st.session_state.get("_loaded_dfs", {})
    for name in dataset_names:
        if name in loaded:
            namespace[name] = loaded[name]
    return namespace


def save_new_variables(namespace: dict, dataset_names: list) -> None:
    """Keep simple variables the code created so later calls can reuse them."""
    store = st.session_state.setdefault("_persistent_vars", {})
    for key, value in namespace.items():
        if key.startswith("_") or key in dataset_names or callable(value):
            continue
        if isinstance(value, SAFE_TYPES):
            store[key] = value


def save_chart(fig) -> None:
    """Store the Plotly figure on disk and queue it for display under the answer."""
    fname = f"{uuid.uuid4().hex}.pkl"
    with open(os.path.join(IMAGE_DIR, fname), "wb") as f:
        pickle.dump(fig, f)
    st.session_state.setdefault("pending_images", []).append(fname)


def pdf_is_valid(path: str) -> bool:
    """A real report is bigger than a tiny stub and has at least one page."""
    if not os.path.exists(path) or os.path.getsize(path) <= MIN_PDF_BYTES:
        return False
    try:
        return len(pypdf.PdfReader(path).pages) > 0
    except Exception:
        return False


def register_new_pdfs(pdfs_before: set) -> str:
    """Check PDFs created by this run. Queue good ones for download, delete broken ones."""
    notes = ""
    for pdf_file in set(os.listdir(PDF_DIR)) - pdfs_before:
        path = os.path.join(PDF_DIR, pdf_file)
        if pdf_is_valid(path):
            st.session_state.setdefault("pending_pdfs", []).append(pdf_file)
            notes += f"\n[PDF Report generated successfully: {pdf_file}]"
        else:
            if os.path.exists(path):
                os.remove(path)
            notes += f"\n[PDF generation failed: '{pdf_file}' was empty or corrupted.]"
    return notes


def make_python_tool(graph_state: dict):
    dataset_names = list(graph_state.get("input_data_paths", {}))

    @tool
    def complete_python_task(python_code: str, thought: str = "") -> str:
        """Run Python code on the loaded datasets to analyze data, build a Plotly chart (assign it to `fig`) or write a PDF report. Use print() to output results.

        Args:
            python_code: The Python code to run.
            thought: Optional short note on what the code is meant to do.
        """
        namespace = build_namespace(dataset_names)
        pdfs_before = set(os.listdir(PDF_DIR))
        printed = StringIO()

        try:
            with redirect_stdout(printed):
                exec(python_code, namespace)
        except Exception as e:
            # Put the error first so it survives the output cut in call_tools
            message = f"EXECUTION ERROR: {type(e).__name__}: {e}"
            if printed.getvalue().strip():
                message += f"\nOutput before the error:\n{printed.getvalue()}"
            return message

        save_new_variables(namespace, dataset_names)
        output = printed.getvalue()

        if namespace.get("fig") is not None:
            save_chart(namespace["fig"])
            output += "\n[Chart generated successfully.]"

        output += register_new_pdfs(pdfs_before)
        return output.strip() or "Code executed successfully."

    return complete_python_task
