"""
rag_pipeline.py

End-to-end Retrieval-Augmented Generation (RAG) pipeline.

This module ties together document loading, chunking, embedding, vector storage,
and LLM generation into a single, configurable interface.

Example:
    from src.pipelines.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline.from_config()
    answer = pipeline.query("What is quantum entanglement?")
    print(answer.response)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    question: str
    response: str
    source_documents: List[Document] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""

    def __str__(self) -> str:
        sources = [doc.metadata.get("source", "unknown") for doc in self.source_documents]
        return (
            f"Q: {self.question}\n"
            f"A: {self.response}\n"
            f"Sources: {', '.join(set(sources))}\n"
            f"Latency: {self.latency_ms:.1f}ms"
        )


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline backed by FAISS and OpenAI.

    The pipeline has two main phases:
      1. Ingestion: load documents -> chunk -> embed -> store in FAISS
      2. Query:     embed question -> retrieve top-k chunks -> generate answer
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        index_path: Optional[str] = None,
    ) -> None:
        self.llm_model = llm_model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.index_path = Path(index_path or "./data/faiss_index")

        # Lazy-init; created on first use
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None
        self._chain: Optional[RetrievalQA] = None

        logger.info(
            f"RAGPipeline initialized | model={llm_model} "
            f"chunk_size={chunk_size} top_k={top_k}"
        )

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_dict: Optional[Dict[str, Any]] = None) -> "RAGPipeline":
        """Build a RAGPipeline from the project YAML config."""
        cfg = config_dict or load_config()
        rag_cfg = cfg.get("rag", {})
        llm_cfg = cfg.get("llm", {})
        vs_cfg = cfg.get("vector_store", {})

        return cls(
            llm_model=llm_cfg.get("model_name", "gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0.0),
            chunk_size=rag_cfg.get("chunk_size", 512),
            chunk_overlap=rag_cfg.get("chunk_overlap", 64),
            top_k=rag_cfg.get("top_k", 5),
            index_path=vs_cfg.get("index_path"),
        )

    # ------------------------------------------------------------------
    # Properties (lazy initialisation)
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            logger.debug("Initialising OpenAI embeddings...")
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return self._embeddings

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_directory(
        self,
        data_dir: str,
        glob_pattern: str = "**/*",
        file_extensions: Optional[List[str]] = None,
    ) -> int:
        """
        Load all documents from a directory, chunk them, embed them,
        and persist the FAISS index to disk.

        Args:
            data_dir: Path to the directory containing source documents.
            glob_pattern: Glob pattern passed to DirectoryLoader.
            file_extensions: Whitelist of extensions (e.g. [".pdf", ".txt"]).
                             If None, loads everything.

        Returns:
            Number of chunks indexed.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        logger.info(f"Starting ingestion from {data_dir}")
        documents = self._load_documents(data_dir, glob_pattern, file_extensions)
        chunks = self._split_documents(documents)

        logger.info(f"Building FAISS index from {len(chunks)} chunks...")
        self._vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Persist index so we don't have to re-embed every time
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.index_path))
        logger.info(f"FAISS index saved to {self.index_path}")

        # Rebuild the chain with the new vectorstore
        self._chain = self._build_chain()
        return len(chunks)

    def load_index(self) -> None:
        """Load a previously persisted FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found at {self.index_path}. "
                "Run ingest_directory() first."
            )
        logger.info(f"Loading FAISS index from {self.index_path}")
        self._vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._chain = self._build_chain()
        logger.info("FAISS index loaded successfully")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: Natural language question.

        Returns:
            RAGResponse with the answer and source documents.

        Raises:
            RuntimeError: If the index has not been loaded or built yet.
        """
        if self._chain is None:
            # Try to load a persisted index automatically
            logger.warning("Chain not initialised. Attempting to load persisted index.")
            self.load_index()

        logger.debug(f"Querying RAG pipeline: {question!r}")
        t0 = time.perf_counter()
        result = self._chain.invoke({"query": question})
        latency_ms = (time.perf_counter() - t0) * 1000

        response = RAGResponse(
            question=question,
            response=result["result"],
            source_documents=result.get("source_documents", []),
            latency_ms=latency_ms,
            model=self.llm_model,
        )
        logger.info(f"RAG query answered in {latency_ms:.1f}ms")
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_documents(
        self,
        data_dir: Path,
        glob_pattern: str,
        file_extensions: Optional[List[str]],
    ) -> List[Document]:
        """Load documents using appropriate loaders for each file type."""
        loader_map = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
        }

        documents: List[Document] = []
        paths = list(data_dir.glob(glob_pattern))

        for path in paths:
            suffix = path.suffix.lower()
            if file_extensions and suffix not in file_extensions:
                continue
            loader_cls = loader_map.get(suffix, TextLoader)
            try:
                loader = loader_cls(str(path))
                docs = loader.load()
                documents.extend(docs)
                logger.debug(f"Loaded {len(docs)} pages from {path.name}")
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Could not load {path.name}: {exc}")

        logger.info(f"Loaded {len(documents)} documents total")
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(
            f"Split {len(documents)} documents into {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _build_chain(self) -> RetrievalQA:
        """Build the LangChain RetrievalQA chain."""
        llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            streaming=False,
        )
        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        logger.debug("RetrievalQA chain built")
        return chain
