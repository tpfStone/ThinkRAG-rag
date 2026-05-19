import streamlit as st
from server.stores.config_store import CONFIG_STORE
from frontend.state import create_llm_instance
from config import RESPONSE_MODE, THINKRAG_PARSE_PROVIDER

st.header("Advanced settings")
advanced_settings = st.container(border=True)

def change_top_k():
    st.session_state["current_llm_settings"]["top_k"] = st.session_state["top_k"]
    CONFIG_STORE.put(key="current_llm_settings", val=st.session_state["current_llm_settings"])
    create_llm_instance()

def change_temperature():
    st.session_state["current_llm_settings"]["temperature"] = st.session_state["temperature"]
    CONFIG_STORE.put(key="current_llm_settings", val=st.session_state["current_llm_settings"])
    create_llm_instance()

def change_system_prompt():
    st.session_state["current_llm_settings"]["system_prompt"] = st.session_state["system_prompt"]
    CONFIG_STORE.put(key="current_llm_settings", val=st.session_state["current_llm_settings"])
    create_llm_instance()

def change_response_mode():
    st.session_state["current_llm_settings"]["response_mode"] = st.session_state["response_mode"]
    CONFIG_STORE.put(key="current_llm_settings", val=st.session_state["current_llm_settings"])
    create_llm_instance()

def change_strict_mode():
    st.session_state["current_llm_settings"]["strict_mode"] = st.session_state["strict_mode"]
    CONFIG_STORE.put(key="current_llm_settings", val=st.session_state["current_llm_settings"])

def change_refuse_gate():
    st.session_state["current_llm_settings"]["use_refuse_gate"] = st.session_state["use_refuse_gate"]
    CONFIG_STORE.put(key="current_llm_settings", val=st.session_state["current_llm_settings"])

def change_consistency_check():
    st.session_state["current_llm_settings"]["use_consistency_check"] = st.session_state["use_consistency_check"]
    CONFIG_STORE.put(key="current_llm_settings", val=st.session_state["current_llm_settings"])

with advanced_settings:
    col_1, _, col_2 = st.columns([4, 2, 4])
    with col_1:
        st.number_input(
            "Top K",
            min_value=1,
            max_value=100,
            help="The number of most similar documents to retrieve in response to a query.",
            value=st.session_state["current_llm_settings"]["top_k"],
            key="top_k",
            on_change=change_top_k,
        )
    with col_2:
        st.select_slider(
            "Temperature",
            options=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            help="The temperature to use when generating responses. Higher temperatures result in more random responses.",
            value=st.session_state["current_llm_settings"]["temperature"],
            key="temperature",
            on_change=change_temperature,
        )
    st.text_area(
        "System Prompt",
        help="The prompt to use when generating responses. The system prompt is used to provide context to the model.",
        value=st.session_state["current_llm_settings"]["system_prompt"],
        key="system_prompt",
        height=240,
        on_change=change_system_prompt,
    )
    st.selectbox(
        "Response Mode",
        options=RESPONSE_MODE,
        help="Sets the Llama Index Query Engine response mode used when creating the Query Engine. Default: `compact`.",
        key="response_mode",
        index=RESPONSE_MODE.index(st.session_state["current_llm_settings"]["response_mode"]), # simple_summarize by default
        on_change=change_response_mode,
    )
    st.subheader("RAG pipeline")
    pipeline_col_1, pipeline_col_2, pipeline_col_3 = st.columns(3)
    with pipeline_col_1:
        st.checkbox(
            "Strict Prompt",
            help="Ask the model to answer only from retrieved evidence.",
            value=st.session_state["current_llm_settings"].get("strict_mode", True),
            key="strict_mode",
            on_change=change_strict_mode,
        )
    with pipeline_col_2:
        st.checkbox(
            "Refusal Gate",
            help="Block answers when retrieval relevance is too weak or ambiguous.",
            value=st.session_state["current_llm_settings"].get("use_refuse_gate", True),
            key="use_refuse_gate",
            on_change=change_refuse_gate,
        )
    with pipeline_col_3:
        st.checkbox(
            "Consistency Check",
            help="Check whether the generated answer is supported by retrieved evidence.",
            value=st.session_state["current_llm_settings"].get("use_consistency_check", True),
            key="use_consistency_check",
            on_change=change_consistency_check,
        )
    st.caption(f"Document parse provider: `{THINKRAG_PARSE_PROVIDER}`")

# For debug purpost only
def show_session_state():
    st.write("")
    with st.expander("List of current application parameters"):
        state = dict(sorted(st.session_state.items()))
        st.write(state)

# show_session_state()
