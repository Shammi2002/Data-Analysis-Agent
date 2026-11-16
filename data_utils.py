"""Shared state types and the dataset summaries the model sees each turn."""
import operator
from dataclasses import dataclass
from typing import Annotated, List, Sequence, TypedDict

import polars as pl
import streamlit as st
from langchain_core.messages import BaseMessage


@dataclass
class InputData:
    variable_name: str
    data_path: str
    data_description: str


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    input_data_paths: dict        # {variable_name: (file_path, description)}
    output_image_paths: List[str]


def init_session_state() -> None:
    """Create the session keys the agent relies on, if they are missing."""
    st.session_state.setdefault("_persistent_vars", {})
    st.session_state.setdefault("_loaded_dfs", {})


def describe_df(df: pl.DataFrame) -> str:
    lines = [f"Shape: {df.height} rows x {df.width} columns\n"]
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].null_count()
        if dtype.is_numeric() and df[col].drop_nulls().len() > 0:
            lines.append(
                f"  - {col} ({dtype}): "
                f"min={df[col].min():.2f}, "
                f"max={df[col].max():.2f}, "
                f"mean={df[col].mean():.2f}, "
                f"nulls={nulls}"
            )
        else:
            sample = df[col].drop_nulls().head(3).to_list()
            unique = df[col].n_unique()
            lines.append(
                f"  - {col} ({dtype}): "
                f"sample={sample}, "
                f"unique={unique}, "
                f"nulls={nulls}"
            )
    return "\n".join(lines)


def create_data_summary(state: AgentState) -> str:
    summary = "Datasets ALREADY loaded as variables:\n"
    loaded_dfs = st.session_state.get("_loaded_dfs", {})
    for var_name, (path, desc) in state.get("input_data_paths", {}).items():
        df = loaded_dfs.get(var_name)
        summary += f"\nVariable: `{var_name}`\n"
        if df is not None:
            summary += describe_df(df)
        else:
            summary += f"  {desc}\n"
    return summary
