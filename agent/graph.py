"""The LangGraph loop: agent (LLM) -> tools -> agent ... until the model stops calling tools."""
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from agent.tools import make_python_tool
from config import MAX_HISTORY_MESSAGES, MAX_TOOL_OUTPUT
from data_utils import AgentState, create_data_summary
from llm import get_llm
from prompts import build_system_prompt

CHAT_TYPES = (HumanMessage, AIMessage, ToolMessage)


def trim_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Keep only the most recent messages so old conversation doesn't pile up forever.

    The cut is moved forward to a user message, so the history never starts with a
    tool result whose tool call was cut off (the API rejects that).
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return list(messages)
    trimmed = list(messages[-MAX_HISTORY_MESSAGES:])
    while trimmed and not isinstance(trimmed[0], HumanMessage):
        trimmed.pop(0)
    return trimmed or [messages[-1]]


def _fallback_text(messages: List[BaseMessage]) -> str:
    """Text for a final answer that came back empty: show the latest tool output, if any."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and str(m.content).strip():
            return f"Here is the result of the analysis:\n\n```\n{m.content}\n```"
    return "I couldn't produce an answer for that. Please try rephrasing the question."


def call_model(state: AgentState):
    summary = create_data_summary(state)
    system_msg = SystemMessage(content=build_system_prompt(summary))
    conversation = trim_history([m for m in state["messages"] if isinstance(m, CHAT_TYPES)])
    python_tool = make_python_tool(dict(state))

    response = None
    last_error = None
    for _attempt in range(2):  # retry once, Groq sometimes returns "tool_use_failed"
        try:
            model_with_tools = get_llm().bind_tools([python_tool], tool_choice="auto")
            response = model_with_tools.invoke([system_msg] + conversation)
            break
        except Exception as e:
            last_error = e
            if "tool_use_failed" not in str(e):
                break

    if response is None:
        return {"messages": [AIMessage(content=f"Model invocation failed: {last_error}")]}

    # A final answer (no tool calls) with no text would show nothing in the chat
    if not response.tool_calls and not str(response.content).strip():
        response = AIMessage(content=_fallback_text(state["messages"]), id=response.id)

    return {"messages": [response]}


def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        python_tool = make_python_tool(dict(state))
        tool_registry = {python_tool.name: python_tool}

        for tool_call in last_message.tool_calls:
            tool_obj = tool_registry.get(tool_call["name"])
            if tool_obj:
                try:
                    message_content = str(tool_obj.invoke(tool_call["args"]))
                except Exception as e:
                    message_content = f"Tool error: {e}"
            else:
                message_content = f"Tool '{tool_call['name']}' not found."

            truncated = message_content[:MAX_TOOL_OUTPUT]
            if len(message_content) > MAX_TOOL_OUTPUT:
                truncated += "\n[...output truncated for brevity]"

            tool_messages.append(ToolMessage(
                content=truncated,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            ))

    return {"messages": tool_messages}


def route_to_tools(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_conditional_edges("agent", route_to_tools)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    return workflow.compile()
