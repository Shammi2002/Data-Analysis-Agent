"""Keeps the conversation and links each answer to the charts and PDFs it produced."""
import uuid
from typing import List

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.graph import build_graph
from config import RECURSION_LIMIT
from data_utils import InputData


class PythonChatbot:
    def __init__(self):
        self.chat_history: List[BaseMessage] = []
        self.output_image_paths: dict = {}  # {ai_message_id: [chart .pkl names]}
        self.output_pdf_paths: dict = {}    # {ai_message_id: [pdf names]}
        self.graph = build_graph()

    def user_sent_message(self, user_query: str, input_data: List[InputData]):
        self.chat_history.append(HumanMessage(content=user_query))

        input_data_paths = {d.variable_name: (d.data_path, d.data_description) for d in input_data}

        st.session_state["pending_images"] = []
        st.session_state["pending_pdfs"] = []

        input_state = {
            "messages": self.chat_history,
            "input_data_paths": input_data_paths,
            "output_image_paths": [],
        }

        try:
            result = self.graph.invoke(input_state, config={"recursion_limit": RECURSION_LIMIT})
        except GraphRecursionError:
            result = {"messages": self.chat_history + [
                AIMessage(content="I wasn't able to finish this within the step limit.")]}
        except Exception as e:
            result = {"messages": self.chat_history + [
                AIMessage(content=f"Something went wrong: {e}")]}

        self.chat_history = [
            m for m in result["messages"] if isinstance(m, (HumanMessage, AIMessage, ToolMessage))
        ]

        # Attach this turn's charts and PDFs to the final answer
        ai_messages = [m for m in self.chat_history if isinstance(m, AIMessage)]
        if ai_messages:
            last_ai = ai_messages[-1]
            if not str(last_ai.content).strip() and not last_ai.tool_calls:
                last_ai.content = "Done."
            if not last_ai.id:
                last_ai.id = str(uuid.uuid4())

            new_images = list(st.session_state.get("pending_images", []))
            if new_images:
                self.output_image_paths[last_ai.id] = new_images

            new_pdfs = list(st.session_state.get("pending_pdfs", []))
            if new_pdfs:
                self.output_pdf_paths[last_ai.id] = new_pdfs

        live_ids = {m.id for m in self.chat_history}
        self.output_image_paths = {k: v for k, v in self.output_image_paths.items() if k in live_ids}
        self.output_pdf_paths = {k: v for k, v in self.output_pdf_paths.items() if k in live_ids}

    def reset_chat(self):
        st.session_state["_persistent_vars"] = {}
        st.session_state["_loaded_dfs"] = {}
        st.session_state["selected_files"] = []
        st.session_state["pending_images"] = []
        st.session_state["pending_pdfs"] = []
        self.chat_history = []
        self.output_image_paths = {}
        self.output_pdf_paths = {}
