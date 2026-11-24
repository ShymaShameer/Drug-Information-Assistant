# for cleaner output suppressing the log messages
import logging
import re
import shutil
import threading
from pathlib import Path

# Silence HTTP request logs from httpx and urllib3
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Silence internal LlamaIndex verbose logs
logging.getLogger("llama_index").setLevel(logging.WARNING)


from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage
 )


from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer,ChatSummaryMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.readers.file import PyMuPDFReader # for advanced PDF parsing
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import TransformQueryEngine # Query rewriting
from llama_index.core.indices.query.query_transform import HyDEQueryTransform 
from llama_index.core.retrievers import TransformRetriever
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator,FilterCondition
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    RERANKER_TOP_N,
    RERANKER_MODEL_NAME,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
    BEST_RERANKER_STRATEGY,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm,
    #initialise_hyde_llm
)    

# ==========================================
# Drug / year metadata helpers
# ==========================================
# Files are named BRAND_generic_YEAR.pdf (e.g. FARXIGA_dapagliflozin_2018.pdf).
# The BRAND is the filter key, because different brands can share one generic
# (Victoza and Saxenda are both liraglutide; Ozempic, Wegovy and Rybelsus are semaglutide).
_YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# Dropdown placeholders that mean "no filter"
_ALL_VALUES = {"", "all", "any", "none", "all drugs", "all years", "all years (timeline)"}


def _resolve_drug(text: str | None) -> str | None:
    """Dropdown label -> brand key. 'FARXIGA (dapagliflozin)' -> 'farxiga'.
    Placeholders such as 'All Drugs' -> None (= no drug filter)."""
    if not text:
        return None
    cleaned = text.split("(")[0].strip().lower()
    return None if cleaned in _ALL_VALUES else cleaned


def _resolve_year(text: str | None) -> int | None:
    """Extract a 4-digit year (2000-2099) from a string, or None
    (so 'All Years (Timeline)' and 'Default' mean 'no year filter')."""
    if not text:
        return None
    match = _YEAR_PATTERN.search(str(text))
    return int(match.group(1)) if match else None


def _file_metadata(file_path: str) -> dict:
    """Metadata attached to every page/chunk of a PDF at index time."""
    path = Path(file_path)
    stem = path.stem
    parts = stem.split("_", 2)          # same split as the Streamlit sidebar
    if len(parts) == 3:
        brand, generic, year_label = parts
    else:
        brand, generic, year_label = stem, "", stem

    year = _resolve_year(year_label)
    if year is None:
        print(f"[metadata warning] No year found in '{path.name}'")

    return {
        "file_name": path.name,
        "file_path": str(path),
        "drug": brand.strip().lower(),       # filter key, e.g. "farxiga"
        "generic": generic.strip().lower(),  # e.g. "dapagliflozin"
        "year": year if year is not None else 0,
    }


def _build_filters(
    selected_drug: str | None,
    selected_year: int | str | None,
    selected_files: str | list[str] | None,
) -> MetadataFilters | None:
    """Build a native pre-retrieval metadata filter.

    Every part is optional: drug only, year only, both, or neither.
    Values like None, "", "All" or "Any" simply mean "no filter on that field".

    Priority:
      1. selected_drug / selected_year (explicit, recommended)
      2. selected_files (legacy) -> drug/year are parsed from the filenames
    """
    drug: str | None = None
    years: list[int] = []

    if selected_drug or selected_year:
        # --- 1. Explicit drug / year (tolerant of "All", None, "2018", 2018) ---
        if selected_drug:
            drug = _resolve_drug(str(selected_drug))  # None for "All" -> no drug filter
        if selected_year:
            year = _resolve_year(str(selected_year))  # None for "All" -> no year filter
            if year:
                years = [year]
        drugs = [drug] if drug else []

    elif selected_files:
        # --- 2. Legacy: derive drug(s)/year(s) from the filenames ---
        file_list = selected_files if isinstance(selected_files, list) else [selected_files]
        metas = [_file_metadata(f) for f in file_list]
        drugs = sorted({m["drug"] for m in metas if m["drug"] != "unknown"})
        years = sorted({m["year"] for m in metas if m["year"]})

    else:
        return None

    conditions = []
    if len(drugs) == 1:
        conditions.append(MetadataFilter(key="drug", value=drugs[0], operator=FilterOperator.EQ))
    elif len(drugs) > 1:
        conditions.append(MetadataFilter(key="drug", value=drugs, operator=FilterOperator.IN))

    if len(years) == 1:
        conditions.append(MetadataFilter(key="year", value=years[0], operator=FilterOperator.EQ))
    elif len(years) > 1:
        conditions.append(MetadataFilter(key="year", value=years, operator=FilterOperator.IN))

    if not conditions:
        return None  # nothing to filter on -> search the whole index

    return MetadataFilters(filters=conditions, condition=FilterCondition.AND)


def _create_new_vector_store(
        embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""

    print(
        "Creating new vector store from all files in the 'data' directory..."
    )

# Use PyMuPDFReader for robust extraction of PDF text and image descriptions/metadata
    file_extractor = {".pdf": PyMuPDFReader()}

    # To read all the files in the specified directory.
    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        required_exts=[".pdf"],          # Ensures only PDFs are loaded
        file_extractor=file_extractor,
        file_metadata=_file_metadata,    # <-- tags every page with drug + year
        recursive=True
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # To break the long document into smaller, more manageable chunks.
    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # This is the core of the vector store. It takes the text chunks,
    # uses the embedding model to convert them to vectors, and stores them.
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model
    )

    # To save the newly created index to disk for future use.
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index


def _index_has_drug_year_metadata(index: VectorStoreIndex) -> bool:
    """True if the loaded index was built with the 'drug' and 'year' metadata."""
    for node in index.docstore.docs.values():
        return all(k in node.metadata for k in ("drug", "generic", "year"))
    return False


def _load_or_build_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """

    # Create the parent directory if it doesn't exist.
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Check if the directory contains any files.
    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        # We must provide the embed_model when loading the index.
        index = load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )

        # An index built before this change has no drug/year metadata,
        # so every filter would return nothing -> rebuild it once.
        if not _index_has_drug_year_metadata(index):
            print("Existing index has no drug/year metadata. Rebuilding...")
            for item in VECTOR_STORE_PATH.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            return _create_new_vector_store(embed_model)

        return index
    else:
        # If the directory is empty,
        # call our internal function to build the index.
        return _create_new_vector_store(embed_model)


# Only one thread may load/build the index at a time. Streamlit reruns (file save, widget
# change, second browser tab) can start a second script run while the first is still embedding;
# two threads embedding at once crashes PyTorch on Apple GPUs (MPS) and builds the index twice.
_INDEX_LOCK = threading.Lock()


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Thread-safe: loads the vector store from disk, or builds it once if missing/outdated."""
    with _INDEX_LOCK:
        return _load_or_build_vector_store(embed_model)


# ==========================================
# Compare-years mode
# ==========================================
COMPARE_YEARS_ADDENDUM = """

COMPARE MODE
- The context contains label excerpts from SEVERAL label years of the same drug.
  Each excerpt starts with a header like "[DRUG | USPI label YEAR]".
- Compare the years explicitly: what is present in a later label but not in an earlier one
  (added), what is in an earlier one but not in a later one (removed or reworded), and what is
  unchanged. Name the year for every statement.
- Only claim that something was added or removed if the excerpts support it. If an excerpt for
  a year does not mention a topic, say "not found in the retrieved [year] excerpts"; do not
  conclude that it is absent from that full label.
- Prefer the "Recent Major Changes" sections when they are present.
"""


class CompareYearsRetriever(BaseRetriever):
    """Retrieves (and reranks) separately inside each selected label year, so every
    year is represented in the context. A single shared top-k could be dominated by one year."""

    def __init__(self, retrievers: dict[int, BaseRetriever], reranker, drug_label: str):
        self._retrievers = retrievers      # {year: retriever already filtered to drug + year}
        self._reranker = reranker
        self._drug_label = drug_label
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        combined: list[NodeWithScore] = []
        for year, retriever in sorted(self._retrievers.items()):   # oldest year first
            nodes = retriever.retrieve(query_bundle)
            nodes = self._reranker.postprocess_nodes(nodes, query_bundle)  # top_n per year
            for n in nodes:
                meta = {**n.node.metadata, "drug": self._drug_label, "year": year}
                tagged = TextNode(
                    text=f"[{self._drug_label.upper()} | USPI label {year}]\n{n.node.get_content()}",
                    metadata=meta,                                  # keeps file_name / page for the UI
                    excluded_llm_metadata_keys=list(meta.keys()),   # the header already tells the LLM
                    excluded_embed_metadata_keys=list(meta.keys()),
                )
                combined.append(NodeWithScore(node=tagged, score=n.score))
        return combined


def get_chat_engine(
    llm: Groq,
    embed_model: HuggingFaceEmbedding,
    similarity_top_k: int = 10,
    system_prompt: str | None = None,
    selected_files: str | list[str] | None = None,  # legacy: drug/year parsed from filename
    selected_drug: str | None = None,               # e.g. "FARXIGA (dapagliflozin)" or "dapagliflozin"
    selected_year: int | str | None = None,         # e.g. 2018
    compare_years: list[int] | None = None,         # e.g. [2016, 2018] -> compare mode (needs selected_drug)
    vector_index: VectorStoreIndex | None = None,   # pass an already-loaded index to reuse it
) -> CondensePlusContextChatEngine:
    """Initialises and returns the main conversational RAG chat engine with HyDE and
    pre-retrieval drug / year metadata filtering. Optionally compares several label years."""

    if vector_index is None:
        vector_index = get_vector_store(embed_model)

    reranker = SentenceTransformerRerank(
        top_n=RERANKER_TOP_N,
        model=RERANKER_MODEL_NAME
    )

    # 2a. Use the dynamic system_prompt if passed, otherwise fall back to LLM_SYSTEM_PROMPT
    active_system_prompt = system_prompt if system_prompt is not None else LLM_SYSTEM_PROMPT

    if compare_years:
        # ---- Compare mode: one filtered + reranked retrieval per year ----
        years = sorted({int(y) for y in compare_years})
        if len(years) < 2:
            raise ValueError("Compare mode needs at least two different years.")
        drug = _resolve_drug(selected_drug or "")
        if drug is None:
            raise ValueError("Compare mode needs a valid selected_drug.")

        per_year_retrievers = {
            year: vector_index.as_retriever(
                similarity_top_k=similarity_top_k,
                filters=_build_filters(drug, year, None),
            )
            for year in years
        }
        base_retriever: BaseRetriever = CompareYearsRetriever(per_year_retrievers, reranker, drug)
        postprocessors = []   # reranking already done per year inside the retriever
        active_system_prompt += COMPARE_YEARS_ADDENDUM
    else:
        # ---- Single-label mode ----
        # 1. Native metadata filter -> applied BEFORE the top-k search,
        #    so top-k is chosen only among chunks of the selected drug/year.
        filters = _build_filters(selected_drug, selected_year, selected_files)

        ## 1a. Use the dynamic similarity_top_k parameter from Streamlit sidebar
        base_retriever = vector_index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=filters,
        )
        postprocessors = [reranker]

    # 'include_original=True' ensures the system searches with BOTH
    # the hallucinated answer and the user's real question.
    hyde = HyDEQueryTransform(
        include_original=True,
        llm=llm,
    )

    # Query transformation wrapper (HyDE runs once, then feeds every per-year retriever)
    hyde_retriever = TransformRetriever(
        retriever=base_retriever,
        query_transform=hyde
    )

    memory = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT,
        llm=llm, 
    )    

    chat_engine = CondensePlusContextChatEngine(
        retriever=hyde_retriever,
        llm=llm,
        memory=memory,
        system_prompt=active_system_prompt,
        node_postprocessors=postprocessors
    )
   
    return chat_engine