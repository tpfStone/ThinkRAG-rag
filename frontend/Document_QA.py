# Document-based Q&A
from pathlib import Path
import time
import re
import streamlit as st
import pandas as pd
import yaml
from server.stores.chat_store import CHAT_MEMORY
from llama_index.core.llms import ChatMessage, MessageRole
from server.engine import create_query_engine
from server.stores.config_store import CONFIG_STORE

def perform_pipeline_query(prompt):
    return st.session_state.rag_pipeline.answer(prompt)

def perform_query(prompt):
    if st.session_state.get("rag_pipeline") is not None:
        try:
            return perform_pipeline_query(prompt)
        except Exception as e:
            print(f"Pipeline query failed: {type(e).__name__}: {e}")
            st.session_state.rag_pipeline = None
            st.warning("RAGPipeline failed. Falling back to the original query engine.")

    if not st.session_state.query_engine:
        print("Index is not initialized yet")
    if (not prompt) or prompt.strip() == "":
        print("Query text is required")
    try:
        query_response = st.session_state.query_engine.query(prompt)
        return query_response
    except Exception as e:
        # print(f"An error occurred while processing the query: {e}")
        print(f"An error occurred while processing the query: {type(e).__name__}: {e}")

# https://github.com/halilergul1/QA-app
def simple_format_response_and_sources(response):
    primary_response = getattr(response, 'response', '')
    output = {"response": primary_response}
    sources = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            node_data = getattr(node, 'node', None)
            if node_data:
                metadata = getattr(node_data, 'metadata', {})
                text = getattr(node_data, 'text', '')
                text = re.sub(r'\n\n|\n|\u2028', lambda m: {'\n\n': '\u2028', '\n': ' ', '\u2028': '\n\n'}[m.group()], text)
                source_info = {
                    "file": metadata.get('file_name', 'N/A'),
                    "page": metadata.get('page_label', 'N/A'),
                    "text": text
                }
                sources.append(source_info)
    output['sources'] = sources
    return output

def initialize_rag_pipeline(index):
    try:
        from src.rag.pipeline import RAGPipeline

        config_path = Path(__file__).resolve().parents[1] / "configs" / "C4_full.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            pipe_cfg = yaml.safe_load(f)
        st.session_state.rag_pipeline = RAGPipeline(pipe_cfg, index=index)
        print("RAGPipeline initialized with configs/C4_full.yaml")
    except Exception as e:
        st.session_state.rag_pipeline = None
        print(f"RAGPipeline initialization skipped: {type(e).__name__}: {e}")

def render_confidence(result):
    if result.get("refused"):
        st.warning(f"Refused: {result.get('refusal_reason', 'unknown')}")
        return

    label = result.get("consistency_label", "N/A")
    reason = result.get("consistency_reason", "")
    if label == "Y":
        st.success(f"Confidence Y - {reason}")
    elif label == "P":
        st.warning(f"Confidence P - {reason}")
    elif label == "N":
        st.error(f"Confidence N - {reason}")
    else:
        st.info("Confidence N/A")

def render_pipeline_response(response, query_time):
    answer = response.get("answer", "")
    st.write(answer)
    render_confidence(response)
    st.write(f"Took {query_time} second(s)")

    details = response.get("source_details", [])
    details_title = f"Found {len(details)} document(s), max score {response.get('max_score', 0.0):.3f}"
    with st.expander(details_title, expanded=False):
        if details:
            source_nodes = []
            for item in details:
                text = item.get("text", "")
                short_text = text[:80] + "..." if len(text) > 80 else text
                source_nodes.append(
                    {
                        "Title": item.get("file", "N/A"),
                        "Page": item.get("page", "N/A"),
                        "Text": short_text,
                        "Score": f"{item.get('score', 0.0):.3f}",
                    }
                )
            st.table(pd.DataFrame(source_nodes))
        else:
            st.write("No source details.")

    return answer

def chatbox():

    # Load Q&A history
    messages = CHAT_MEMORY.get() 
    if len(messages) == 0:
        # Initialize Q&A record
        CHAT_MEMORY.put(ChatMessage(role=MessageRole.ASSISTANT, content="Feel free to ask about anything in the knowledge base"))
        messages = CHAT_MEMORY.get()

    # Show Q&A records
    for message in messages: 
        with st.chat_message(message.role):
            st.write(message.content)

    if prompt := st.chat_input("Input your question"): # Prompt the user to input the question then add it to the message history
        with st.chat_message(MessageRole.USER):
            st.write(prompt)
            CHAT_MEMORY.put(ChatMessage(role=MessageRole.USER, content=prompt))
        with st.chat_message(MessageRole.ASSISTANT):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = perform_query(prompt)
                end_time = time.time()
                query_time = round(end_time - start_time, 2)
                if response is None:
                    st.write("Couldn't come up with an answer.")
                elif isinstance(response, dict):
                    response_text = render_pipeline_response(response, query_time)
                    CHAT_MEMORY.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_text))
                else:
                    try:
                        response_text = st.write_stream(response.response_gen)
                    except TypeError as e:
                        st.warning("Streaming failed due to Ollama/LlamaIndex token count issue. Falling back to non-streaming output.")
                        response_text = str(response)
                        st.write(response_text)
                    except Exception as e:
                        st.warning(f"Streaming failed: {e}. Falling back to non-streaming output.")
                        response_text = str(response)
                        st.write(response_text)
                    st.write(f"Took {query_time} second(s)")
                    details_title = f"Found {len(response.source_nodes)} document(s)"
                    with st.expander(
                            details_title,
                            expanded=False,
                    ):
                        source_nodes = []
                        for item in response.source_nodes:
                            node = item.node
                            score = item.score
                            title = node.metadata.get('file_name', None)
                            if title is None:
                                title = node.metadata.get('title', 'N/A') # if the document is a webpage, use the title
                                continue
                            page_label = node.metadata.get('page_label', 'N/A')
                            text = node.text
                            short_text = text[:50] + "..." if len(text) > 50 else text
                            source_nodes.append({"Title": title, "Page": page_label, "Text": short_text, "Score": f"{score:.2f}"})
                        df = pd.DataFrame(source_nodes)
                        st.table(df)
                    # store the answer in the chat history
                    CHAT_MEMORY.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_text))
def main():
    st.header("Query")
    if st.session_state.llm is not None:
        current_llm_info = CONFIG_STORE.get(key="current_llm_info")
        current_llm_settings = CONFIG_STORE.get(key="current_llm_settings")
        st.caption("LLM `" + current_llm_info["service_provider"] + "` `" + current_llm_info["model"] + 
                   "` Response mode `" + current_llm_settings["response_mode"] + 
                   "` Top K `" + str(current_llm_settings["top_k"]) + 
                   "` Temperature `" + str(current_llm_settings["temperature"]) + 
                   "` Reranking `" + str(current_llm_settings["use_reranker"]) + 
                   "` Top N `" + str(current_llm_settings["top_n"]) + 
                   "` Reranker `" + current_llm_settings["reranker_model"] + "`"
                   )
        if st.session_state.index_manager is not None:
            if st.session_state.index_manager.check_index_exists():
                st.session_state.index_manager.load_index()
                st.session_state.query_engine = create_query_engine(
                    index=st.session_state.index_manager.index, 
                    use_reranker=current_llm_settings["use_reranker"], 
                    response_mode=current_llm_settings["response_mode"], 
                    top_k=current_llm_settings["top_k"],
                    top_n=current_llm_settings["top_n"],
                    reranker=current_llm_settings["reranker_model"])
                initialize_rag_pipeline(st.session_state.index_manager.index)
                print("Index loaded and query engine created")
                chatbox()
            else:
                print("Index does not exist yet")
                st.warning("Your knowledge base is empty. Please upload some documents into it first.")
        else:
            print("IndexManager is not initialized yet.")
            st.warning("Please upload documents into your knowledge base first.")
    else:
        st.warning("Please configure LLM first.")

main()
