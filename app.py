"""Streamlit UI. Run with:  streamlit run app.py"""
import os
import pickle

import polars as pl
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbot import PythonChatbot
from config import IMAGE_DIR, PDF_DIR, UPLOAD_DIR
from data_utils import InputData, init_session_state

st.set_page_config(page_title="Telecom Data Analysis Agent", layout="wide")
init_session_state()


# ---------- helpers ----------
def render_attachments(chatbot: PythonChatbot, msg_id) -> None:
    """Show the charts and PDF download buttons that belong to one answer."""
    for fname in chatbot.output_image_paths.get(msg_id, []):
        fpath = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "rb") as f:
                    fig = pickle.load(f)
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Could not render chart: {e}")

    for pdf_name in chatbot.output_pdf_paths.get(msg_id, []):
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label=f"Download PDF Report: {pdf_name}",
                    data=f.read(),
                    file_name=pdf_name,
                    mime="application/pdf",
                    key=f"download_{msg_id}_{pdf_name}",
                )


def render_history(chatbot: PythonChatbot) -> None:
    for msg in chatbot.chat_history:
        if isinstance(msg, HumanMessage) and msg.content:
            st.chat_message("user").markdown(msg.content)
        elif isinstance(msg, AIMessage):
            has_files = msg.id in chatbot.output_image_paths or msg.id in chatbot.output_pdf_paths
            if not msg.content and not has_files:
                continue  # tool-call step with nothing to show
            with st.chat_message("assistant"):
                if msg.content:
                    st.markdown(msg.content)
                render_attachments(chatbot, msg.id)


def build_input_data() -> list:
    loaded_dfs = st.session_state.get("_loaded_dfs", {})
    input_data_list = []
    for f in st.session_state.get("selected_files", []):
        var_name = f.split(".")[0]
        df = loaded_dfs.get(var_name)
        cols = ", ".join(df.columns) if df is not None else "unknown"
        input_data_list.append(InputData(
            variable_name=var_name,
            data_path=os.path.abspath(os.path.join(UPLOAD_DIR, f)),
            data_description=f"CSV dataset with columns: {cols}",
        ))
    return input_data_list


# ---------- page ----------
st.title("Telecom Data Analysing Agent")

st.subheader("1. Upload New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved '{uploaded_file.name}'")

st.divider()

st.subheader("2. Select Dataset for Analysis")
files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]

if not files:
    st.warning("No CSV files found. Please upload one first.")
else:
    selected_filenames = st.multiselect("Select a file:", files)

    if st.button("Go"):
        if not selected_filenames:
            st.warning("Please select at least one file.")
        else:
            loaded_dfs = {}
            selected_files = []
            for selected_filename in selected_filenames:
                file_path = os.path.join(UPLOAD_DIR, selected_filename)
                try:
                    df = pl.read_csv(file_path)
                    loaded_dfs[selected_filename.split(".")[0]] = df
                    selected_files.append(selected_filename)

                    with st.expander(f"📄 {selected_filename}", expanded=True):
                        st.dataframe(df.head(5).to_pandas())
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rows", f"{df.height:,}")
                        c2.metric("Columns", df.width)
                        c3.metric("File Size", f"{os.path.getsize(file_path) / 1024:.1f} KB")
                except Exception as e:
                    st.error(f"Error loading '{selected_filename}': {e}")

            if loaded_dfs:
                st.session_state["_loaded_dfs"] = loaded_dfs
                st.session_state["selected_files"] = selected_files
                st.session_state["visualization_chatbot"] = PythonChatbot()
                st.success(f"Loaded {len(loaded_dfs)} file(s) successfully.")

st.divider()

if st.session_state.get("_loaded_dfs"):
    if "visualization_chatbot" not in st.session_state:
        st.session_state["visualization_chatbot"] = PythonChatbot()
    chatbot: PythonChatbot = st.session_state["visualization_chatbot"]

    if st.button("🔄 Reset Chat"):
        chatbot.reset_chat()
        st.rerun()

    chat_container = st.container(height=600)
    with chat_container:
        render_history(chatbot)

    if prompt := st.chat_input("Ask me anything about your data, or request a PDF summary report..."):
        with chat_container:
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chatbot.user_sent_message(prompt, input_data=build_input_data())
        st.rerun()
