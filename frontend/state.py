import streamlit as st
import config as config
from server.models import ollama
from server.models.llm_api import create_openai_llm, check_openai_llm
from server.models.ollama import create_ollama_llm
from server.models.embedding import create_embedding_model
from server.index import IndexManager
from server.stores.config_store import CONFIG_STORE

PIPELINE_SETTING_DEFAULTS = {
    "strict_mode": True,
    "use_refuse_gate": True,
    "use_consistency_check": True,
}

def _local_embedding_model_exists(model_name):
    model_path = config.EMBEDDING_MODEL_PATH.get(model_name)
    if model_path is None or config.MODEL_DIR is None:
        return False

    from pathlib import Path

    return (Path(config.MODEL_DIR) / model_path).exists()

def _normalize_llm_settings(current_llm_settings):
    normalized_settings = current_llm_settings.copy()
    settings_changed = False

    for key, value in PIPELINE_SETTING_DEFAULTS.items():
        if key not in normalized_settings:
            normalized_settings[key] = value
            settings_changed = True

    embedding_model = normalized_settings.get("embedding_model", config.DEFAULT_EMBEDDING_MODEL)
    uses_local_embedding = embedding_model in config.EMBEDDING_MODEL_PATH and embedding_model != config.ALIYUN_EMBEDDING_MODEL

    if uses_local_embedding and not _local_embedding_model_exists(embedding_model):
        normalized_settings["embedding_model"] = config.DEFAULT_EMBEDDING_MODEL
        settings_changed = True
        print(
            f"Embedding model migrated from {embedding_model} to "
            f"{config.DEFAULT_EMBEDDING_MODEL}; local model files were not found."
        )

    if settings_changed:
        CONFIG_STORE.put(key="current_llm_settings", val=normalized_settings)

    return normalized_settings

def find_api_by_model(model_name):
    for api_name, api_info in config.LLM_API_LIST.items():
        if model_name in api_info['models']:
            return api_info

def _mask_current_llm_info(current_llm_info):
    if current_llm_info is None:
        return None
    masked = current_llm_info.copy()
    if masked.get("api_key"):
        masked["api_key"] = "<configured>"
    return masked

def _api_key_from_runtime(sp):
    return st.session_state.get(sp + "_api_key") or config.LLM_API_LIST[sp].get("api_key", "")

def _current_llm_info_from_state():
    sp = st.session_state.get("llm_service_provider_selected")
    if sp is None:
        return None

    if sp == "Ollama":
        model = st.session_state.get("ollama_model_selected")
        if model is None:
            return None
        return {
            "service_provider": sp,
            "model": model,
        }

    if sp not in config.LLM_API_LIST:
        return None

    model_key = sp + "_model_selected"
    base_key = sp + "_api_base"
    api_key = sp + "_api_key"
    valid_key = api_key + "_valid"

    model = st.session_state.get(model_key)
    api_base = st.session_state.get(base_key, config.LLM_API_LIST[sp]["api_base"])
    api_key_value = _api_key_from_runtime(sp)
    api_key_valid = st.session_state.get(valid_key, False)

    if model is None or api_base in (None, "") or api_key_value in (None, ""):
        return None

    return {
        "service_provider": sp,
        "model": model,
        "api_base": api_base,
        "api_key_valid": api_key_valid,
    }

def _refresh_current_llm_info_from_state(current_llm_info):
    state_llm_info = _current_llm_info_from_state()
    if state_llm_info is None:
        return current_llm_info

    if current_llm_info != state_llm_info:
        CONFIG_STORE.put(key="current_llm_info", val=state_llm_info)
        return state_llm_info

    return current_llm_info

# Initialize st.session_state
def init_keys():

    # Initialize LLM
    if "llm" not in st.session_state.keys():
        st.session_state.llm = None

    # Initialize index
    if "index_manager" not in st.session_state.keys():
        st.session_state.index_manager = IndexManager(config.DEFAULT_INDEX_NAME)

    # Initialize model selection
    if "ollama_api_url" not in st.session_state.keys():
        st.session_state.ollama_api_url = config.OLLAMA_API_URL

    if "ollama_models" not in st.session_state.keys():
        ollama.get_model_list()
        # Ensure ollama_models is always initialized, even if empty
        if "ollama_models" not in st.session_state.keys():
            st.session_state.ollama_models = []
    
    if "ollama_model_selected" not in st.session_state.keys():
        if (st.session_state.ollama_models is not None and len(st.session_state.ollama_models) > 0):
            st.session_state.ollama_model_selected = st.session_state.ollama_models[0]
            create_ollama_llm(st.session_state.ollama_model_selected)
        else:
            st.session_state.ollama_model_selected = None
    if "llm_api_list" not in st.session_state.keys():
        st.session_state.llm_api_list = [model for api in config.LLM_API_LIST.values() for model in api['models']]
    if "llm_api_selected" not in st.session_state.keys():
        st.session_state.llm_api_selected = st.session_state.llm_api_list[0]
        if st.session_state.ollama_model_selected is None:
            api_object = find_api_by_model(st.session_state.llm_api_selected)
            create_openai_llm(st.session_state.llm_api_selected, api_object['api_base'], api_object['api_key'])
    
    # Initialize query engine
    if "query_engine" not in st.session_state.keys():
        st.session_state.query_engine = None

    if "rag_pipeline" not in st.session_state.keys():
        st.session_state.rag_pipeline = None

    if "system_prompt" not in st.session_state.keys():
        st.session_state.system_prompt = "Chat with me!"
        
    if "response_mode" not in st.session_state.keys():
        response_mode_result = CONFIG_STORE.get(key="response_mode")
        if response_mode_result is not None:
            st.session_state.response_mode = response_mode_result["response_mode"]
        else:
            st.session_state.response_mode = config.DEFAULT_RESPONSE_MODE

    if "ollama_endpoint" not in st.session_state.keys():
        st.session_state.ollama_endpoint = "http://localhost:11434"

    if "chunk_size" not in st.session_state.keys():
        st.session_state.chunk_size = config.DEFAULT_CHUNK_SIZE
    
    if "chunk_overlap" not in st.session_state.keys():
        st.session_state.chunk_overlap = config.DEFAULT_CHUNK_OVERLAP

    if "zh_title_enhance" not in st.session_state.keys():
        st.session_state.zh_title_enhance = config.ZH_TITLE_ENHANCE

    if "max_tokens" not in st.session_state.keys():
        st.session_state.max_tokens = 100
    
    if "top_p" not in st.session_state.keys():
        st.session_state.top_p = 1.0

    # contents related to the knowledge base
    if "websites" not in st.session_state:
        st.session_state["websites"] = []

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = None

# Initialize user data
# TODO: supposed to be loaded from database
    st.session_state.user_id = "user_1"
    st.session_state.kb_id = "kb_1"
    st.session_state.kb_name = "My knowledge base"

def init_llm_sp():

    llm_options = list(config.LLM_API_LIST.keys())
    default_provider = "Aliyun" if "Aliyun" in llm_options else llm_options[0]

    # LLM service provider selection
    if "llm_service_provider_selected" not in st.session_state:
        sp = CONFIG_STORE.get(key="llm_service_provider_selected")
        if sp:
            st.session_state.llm_service_provider_selected = sp["llm_service_provider_selected"]
        else:
            st.session_state.llm_service_provider_selected = default_provider

def init_ollama_endpoint():
    # Initialize Ollama endpoint
    if "ollama_api_url" not in st.session_state.keys():
        ollama_api_url = CONFIG_STORE.get(key="Ollama_api_url")
        if ollama_api_url:
            st.session_state.ollama_api_url = ollama_api_url["Ollama_api_url"]
        else:
            st.session_state.ollama_api_url = config.LLM_API_LIST["Ollama"]["api_base"]

# Initialize llm api model
def init_api_model(sp):
    if sp != "Ollama":
        model_key = sp + "_model_selected"
        if model_key not in st.session_state.keys():
            model_result = CONFIG_STORE.get(key=model_key)
            if model_result:
                st.session_state[model_key] = model_result[model_key]
            else:
                st.session_state[model_key] = config.LLM_API_LIST[sp]["models"][0]


# Initialize llm api base
def init_api_base(sp):
    if sp != "Ollama":
        api_base = sp + "_api_base"
        if api_base not in st.session_state.keys():
            api_key_result = CONFIG_STORE.get(key=api_base)
            if api_key_result is not None:
                st.session_state[api_base] = api_key_result[api_base]
            else:
                st.session_state[api_base] = config.LLM_API_LIST[sp]["api_base"]

# Initialize llm api key
def init_api_key(sp):
    if sp != "Ollama":
        api_key = sp + "_api_key"
        if api_key not in st.session_state.keys():
            api_key_result = CONFIG_STORE.get(key=api_key)
            if api_key_result is not None:
                st.session_state[api_key] = api_key_result[api_key]
                CONFIG_STORE.delete(api_key)
            else:
                st.session_state[api_key] = config.LLM_API_LIST[sp]["api_key"]
        
        valid_key = api_key + "_valid"
        if valid_key not in st.session_state.keys():
            valid_result = CONFIG_STORE.get(key=valid_key)
            if valid_result is None and st.session_state[api_key] not in (None, ""):
                current_base = st.session_state[sp + "_api_base"] if (sp + "_api_base") in st.session_state else config.LLM_API_LIST[sp]["api_base"]
                current_base = current_base.strip().replace("`", "")
                is_valid = check_openai_llm(st.session_state[sp + "_model_selected"], current_base, st.session_state[api_key])
                CONFIG_STORE.put(key=valid_key, val={valid_key: is_valid})
                st.session_state[valid_key] = is_valid
            elif valid_result is None:
                st.session_state[valid_key] = False
                CONFIG_STORE.put(key=valid_key, val={valid_key: False})
            else:
                st.session_state[valid_key] = valid_result[valid_key]

# Initialize LLM settings, like temperature, system prompt, etc.
def init_llm_settings():
    if "current_llm_settings" not in st.session_state.keys():
        current_llm_settings = CONFIG_STORE.get(key="current_llm_settings")
        if current_llm_settings:
            st.session_state.current_llm_settings = _normalize_llm_settings(current_llm_settings)
        else:
            st.session_state.current_llm_settings = {
                "temperature": config.TEMPERATURE,
                "system_prompt": config.SYSTEM_PROMPT,
                "top_k": config.TOP_K,
                "response_mode": config.DEFAULT_RESPONSE_MODE,
                "use_reranker": config.USE_RERANKER,
                "top_n": config.RERANKER_MODEL_TOP_N,
                "embedding_model": config.DEFAULT_EMBEDDING_MODEL,
                "reranker_model": config.DEFAULT_RERANKER_MODEL,
                **PIPELINE_SETTING_DEFAULTS,
            }
            CONFIG_STORE.put(key="current_llm_settings", val=st.session_state.current_llm_settings)


# Create LLM instance if there is related information
def create_llm_instance():
    current_llm_info = CONFIG_STORE.get(key="current_llm_info")
    current_llm_info = _refresh_current_llm_info_from_state(current_llm_info)
    if current_llm_info is not None:
        print("Current LLM info: ", _mask_current_llm_info(current_llm_info))
        if current_llm_info["service_provider"] == "Ollama":
            if ollama.is_alive():
                model_name = current_llm_info["model"]
                st.session_state.llm = ollama.create_ollama_llm(
                    model=model_name, 
                    temperature=st.session_state.current_llm_settings["temperature"],
                    system_prompt=st.session_state.current_llm_settings["system_prompt"],
                )
        else:
            model_name = current_llm_info["model"]
            api_base = current_llm_info["api_base"].strip().replace("`", "")
            api_key = current_llm_info.get("api_key") or _api_key_from_runtime(current_llm_info["service_provider"])
            api_key_valid = current_llm_info["api_key_valid"]
            if api_key_valid and api_key:
                print("LLM instance created successfully.")
                st.session_state.llm = create_openai_llm(
                    model_name=model_name, 
                    api_base=api_base, 
                    api_key=api_key,
                    temperature=st.session_state.current_llm_settings["temperature"],
                    system_prompt=st.session_state.current_llm_settings["system_prompt"],
                )
            else:
                print("Failed to create LLM instance. Please check provider settings / networks / region policy.")
                st.session_state.llm = None
    else:
        print("No current LLM infomation")
        st.session_state.llm = None

def init_state():
    init_keys()
    init_llm_sp()
    init_llm_settings()
    init_ollama_endpoint()
    sp = st.session_state.llm_service_provider_selected
    init_api_model(sp)
    init_api_base(sp)
    init_api_key(sp)
    create_embedding_model(st.session_state["current_llm_settings"]["embedding_model"])
    create_llm_instance()
