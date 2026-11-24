import logging
import streamlit as st
from src.config import DATA_PATH, LLM_SYSTEM_PROMPT

# 1. Silence background HTTP and LlamaIndex logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

from src.model_loader import initialise_llm, get_embedding_model
from src.engine import get_chat_engine, get_vector_store

# --- Page Setup ---
st.set_page_config(
    page_title="FAERS Pharmacovigilance Assistant",
    page_icon="💊",
    layout="wide"
)

# --- Custom Forest Green Theme ---
st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #10251D;
    color: #F1F5F2;
}

/* Main content area */
[data-testid="stMain"] {
    background-color: #10251D;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #19382B;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #19382B;
}

/* Sidebar headings and labels */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #F1F5F2;
}

/* Chat message containers */
[data-testid="stChatMessage"] {
    background-color: #19382B;
    border: 1px solid #28513D;
    border-radius: 12px;
    padding: 16px;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #19382B;
    border: 1px solid #416B53;
    border-radius: 12px;
}

[data-testid="stChatInput"] textarea {
    color: #FFFFFF;
}

/* Text inputs */
.stTextInput input {
    background-color: #19382B;
    color: #FFFFFF;
}

/* Dropdowns */
[data-baseweb="select"] > div {
    background-color: #19382B;
    border-color: #416B53;
    color: #FFFFFF;
}

/* Buttons */
.stButton button {
    background-color: #24543D;
    color: #FFFFFF;
    border: 1px solid #65B887;
    border-radius: 8px;
}

.stButton button:hover {
    background-color: #316B4D;
    color: #FFFFFF;
    border-color: #8AD4A8;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    color: #65B887;
}

/* Expanders (Show Sources) */
[data-testid="stExpander"] {
    background-color: #19382B;
    border: 1px solid #416B53;
    border-radius: 10px;
}

/* Tables */
[data-testid="stTable"],
[data-testid="stDataFrame"] {
    background-color: #19382B;
}

/* Dividers */
hr {
    border-color: #416B53;
}

/* Captions */
.stCaption {
    color: #A9C5B2;
}

/* Links */
a {
    color: #8AD4A8;
}

/* Scrollbar */
::-webkit-scrollbar-thumb {
    background-color: #416B53;
    border-radius: 10px;
}
/* Fix the entire bottom chat input area */
[data-testid="stBottom"] {
    background-color: #10251D !important;
}

[data-testid="stBottom"] > div {
    background-color: #10251D !important;
}

[data-testid="stBottomBlockContainer"] {
    background-color: #10251D !important;
}

/* Chat input outer container */
[data-testid="stChatInput"] {
    background-color: #19382B !important;
    border: 1px solid #416B53 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

/* Inner chat input area */
[data-testid="stChatInput"] > div {
    background-color: #19382B !important;
    border-color: #416B53 !important;
}

/* Text inside input */
[data-testid="stChatInput"] textarea {
    background-color: #19382B !important;
    color: #FFFFFF !important;
}

/* Placeholder text */
[data-testid="stChatInput"] textarea::placeholder {
    color: #A9C5B2 !important;
}
</style>
""", unsafe_allow_html=True)


WELCOME_MESSAGE = (
    "Hello! 👋 Welcome to the Drug Safety Assistant.  \n"
    "You can ask me questions about FAERS data, adverse events "
    "associated with SGLT2 inhibitors and GLP-1 receptor agonists."
)

# --- Engine & Model Caching ---
@st.cache_resource(show_spinner=False)
def load_models():
    llm = initialise_llm()
    embed_model = get_embedding_model()
    return llm, embed_model

@st.cache_resource(show_spinner="Loading vector index (the first run builds it and can take a few minutes)...")
def load_index():
    """Built/loaded exactly once per server process and shared by every engine configuration."""
    _, embed_model = load_models()
    return get_vector_store(embed_model)

@st.cache_resource(show_spinner=False)
def get_cached_chat_engine(
    similarity_top_k: int,
    system_prompt: str,
    selected_drug: str | None,
    selected_year: str | None,
    compare_years: tuple[str, ...] | None,
):
    llm, embed_model = load_models()
    return get_chat_engine(
        llm=llm,
        embed_model=embed_model,
        similarity_top_k=similarity_top_k,
        system_prompt=system_prompt,
        selected_drug=selected_drug,
        selected_year=selected_year,
        compare_years=[int(y) for y in compare_years] if compare_years else None,
        vector_index=load_index(),
    )

def render_sources(sources) -> None:
    """Shows retrieved chunks with file, label year, page and score."""
    with st.expander("🔍 Show Sources"):
        for src_idx, source in enumerate(sources, 1):
            meta = source.metadata
            file_name = meta.get("file_name", "Unknown File")
            year = meta.get("year", "N/A")
            page_num = meta.get("page_label") or meta.get("source", "N/A")
            score = f"{source.score:.4f}" if hasattr(source, "score") and source.score is not None else "N/A"

            st.markdown(
                f"**Source {src_idx}** — File: `{file_name}` | Label year: `{year}` | "
                f"Page: `{page_num}` | Score: `{score}`"
            )
            st.info(source.text)

# --- Sidebar: Configuration Panel & Controls ---

with st.sidebar:

    # Sidebar Logo
    from PIL import Image

    logo = Image.open("assets/Logo image.png")
    st.image(logo,width=400)

    # Sidebar Title
    st.markdown(
        """
        <h2 style="
            text-align: center;
            color: #8AD4A8;
            font-size: 23px;
            margin-top: 5px;
            margin-bottom: 5px;
        ">
            Safety Intelligence Hub
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Dropdowns and configuration controls.

    # 1. Slider to control retrieval top_k
    similarity_top_k = st.slider(
        "Similarity Top K",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of relevant document text chunks to retrieve (per label year in Compare mode)."
    )

    # 2. Dropdown to select system prompt persona
    prompt_preset = st.selectbox(
        "System Prompt Preset",
        options=[
            "Standard Pharmacovigilance Analyst",
            "Concise Safety Summary",
            "Strict Document Verification"
        ],
        index=0
    )

    # 3. Two-Tier Filter: Brand (Generic) -> Label Year. # For USPI labeling
    available_files = sorted([f.name for f in DATA_PATH.glob("*.pdf")])

    # Structure: {"SAXENDA (liraglutide)": [("2014", "SAXENDA_liraglutide_2014.pdf"), ("2017", "SAXENDA_liraglutide_2017.pdf")]}
    drug_catalog: dict[str, list[tuple[str, str]]] = {}
    for fname in available_files:
        stem = fname[:-4] if fname.lower().endswith(".pdf") else fname
        parts = stem.split("_", 2)
        if len(parts) == 3:
            brand, generic, year_label = parts
            display_drug = f"{brand} ({generic})"
        else:
            display_drug = stem
            year_label = "Default"
        drug_catalog.setdefault(display_drug, []).append((year_label, fname))

    drug_options = ["All Drugs"] + sorted(drug_catalog.keys())
    selected_drug = st.selectbox(
        "1. Select Drug (Brand / Generic)",
        options=drug_options,
        index=0,
        help="Separates distinct brand formulations even when they share the same active ingredient."
    )

    # Filters passed to the engine (None = no filter on that field)
    drug_filter: str | None = None
    year_filter: str | None = None
    compare_years: tuple[str, ...] | None = None
    config_error: str | None = None

    if selected_drug != "All Drugs":
        drug_filter = selected_drug
        # Sort versions chronologically
        versions = sorted(drug_catalog[selected_drug], key=lambda x: x[0])
        year_labels = [v[0] for v in versions]
        numeric_years = [y for y in year_labels if y.isdigit()]

        mode = "Single label"
        if len(numeric_years) >= 2:
            mode = st.radio(
                "Mode",
                ["Single label", "Compare years"],
                horizontal=True,
                help="Single label: ask about one label year (or all years together). "
                     "Compare years: retrieve from two label years separately and compare them."
            )

        if mode == "Single label":
            year_choice = st.selectbox(
                "2. Select Label Year / Revision",
                options=["All Years (Timeline)"] + year_labels,
                index=0,
                help="Pick 'All Years' to search across time, or a single year for an exact snapshot."
            )
            if year_choice != "All Years (Timeline)":
                year_filter = year_choice
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                year_a = st.selectbox("Compare from", numeric_years, index=0)
            with col_b:
                year_b = st.selectbox("Compare to", numeric_years, index=len(numeric_years) - 1)
            if year_a == year_b:
                config_error = "Please pick two different label years to compare."
            else:
                compare_years = tuple(sorted([year_a, year_b]))

    prompt_mapping = {
        # Uses the detailed prompt from src/config.py
        "Standard Pharmacovigilance Analyst": LLM_SYSTEM_PROMPT,
        "Concise Safety Summary": (
            "Provide direct, bulleted summaries emphasizing high-level safety signals and key facts."
        ),
        "Strict Document Verification": (
            "Answer strictly using the provided context chunks. If information is missing, explicitly state "
            "that the retrieved document context is insufficient."
        )
    }
    selected_system_prompt = prompt_mapping[prompt_preset]

    # Tell the model which label(s) the user selected
    if compare_years:
        selected_system_prompt += (
            f"\n\nSelected drug: {selected_drug}. Comparing USPI label years {compare_years[0]} and {compare_years[1]}."
        )
    elif drug_filter:
        selected_system_prompt += (
            f"\n\nSelected label: {selected_drug}, "
            f"{'USPI label ' + year_filter if year_filter else 'all available label years'}."
        )

    st.divider()
    new_chat_clicked = st.button("🗑️ New Chat", use_container_width=True)

# --- Main UI Header ---
st.title("💊 Drug Information Assistant")
st.caption("FAERS Pharmacovigilance & Safety Query System")

if config_error:
    st.warning(config_error)
    st.stop()

# Scope indicator
if compare_years:
    st.caption(f"📑 Comparing **{selected_drug}** label years **{compare_years[0]}** and **{compare_years[1]}**")
elif drug_filter:
    st.caption(f"📑 Searching **{selected_drug}** — {'label ' + year_filter if year_filter else 'all label years'}")
else:
    st.caption("📑 Searching all drugs and label years")

# --- Initialize / Retrieve Cached Chat Engine ---
# Must happen after sidebar inputs are collected, and BEFORE new_chat_clicked triggers reset()
chat_engine = get_cached_chat_engine(
    similarity_top_k,
    selected_system_prompt,
    drug_filter,
    year_filter,
    compare_years,
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]

# Start a fresh conversation whenever the drug / year / mode selection changes,
# so answers from a different label never mix in one chat.
config_sig = (drug_filter, year_filter, compare_years)
if st.session_state.get("config_sig") != config_sig:
    chat_engine.reset()
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.config_sig = config_sig

# --- Handle New Chat Reset ---
if new_chat_clicked:
    chat_engine.reset()
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.rerun()

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if stored in message history
        if "sources" in msg and msg["sources"]:
            render_sources(msg["sources"])

        # User Feedback Buttons for Assistant responses
        if msg["role"] == "assistant" and idx > 0:
            col1, col2, _ = st.columns([0.05, 0.05, 0.9])
            with col1:
                st.button("👍", key=f"thumb_up_{idx}")
            with col2:
                st.button("👎", key=f"thumb_down_{idx}")

# --- Handle User Input ---
if prompt := st.chat_input("Ask about FAERS data (e.g., 'What is Invokana?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Antidiabetic drug data..."):
            response = chat_engine.chat(prompt)
            response_text = str(response)
            
            # Extract source chunks from LlamaIndex response object
            source_nodes = getattr(response, "source_nodes", [])

            st.markdown(response_text)

            # Display source chunks for the current turn
            if source_nodes:
                render_sources(source_nodes)

    # Save to history with sources attached
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": source_nodes
    })
    st.rerun()