# ---------------------Imports -----------------------------
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
import re

from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from autogen.agentchat.group.guardrails import Guardrail, RegexGuardrail, GuardrailResult
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function, AssistantAgent
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen.agentchat.group import StringContextCondition
from autogen.agentchat import initiate_group_chat
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.group.targets.transition_target import StayTarget, TerminateTarget
from typing import Any, Dict, List, Optional, Annotated
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os
from autogen.tools.experimental import DeepResearchTool
from dotenv import load_dotenv
import subprocess
from pathlib import Path
import logging
import socket
import shutil
import time

# ===== Path Setup =====
ROOT_DIR = Path("/workspace")
DB_PATH = ROOT_DIR / "Research_and_Documents" / "Vector_Database"

# Ensure database directory exists
DB_PATH.mkdir(parents=True, exist_ok=True)

# Initialize once ====================
embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name="docs", embedding_function=embedder)


def chunk_by_sections(text: str) -> list[str]:
    """
    Split markdown by sections (## headers) and subquestions.
    Each chunk is a complete section or subquestion with its answer.
    """
    # Split on ## headers or "Subquestion X:" patterns
    pattern = r'(?=^## |\nSubquestion \d+:)'
    
    parts = re.split(pattern, text, flags=re.MULTILINE)
    
    # Clean up and filter empty chunks
    chunks = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
    
    return chunks

def chunk_by_sentences(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """
    Split text into sentence-based chunks.
    Groups a few sentences together to preserve some context.
    """
    # Clean up the text - remove excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences using regex
    # Handles: . ! ? followed by space and capital letter (or end of string)
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    # Filter out very short sentences (likely artifacts)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # Group sentences into chunks
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def ingest(file_path: str) -> int:
    """Index a markdown file by sections. Returns chunk count."""
    path = Path(file_path)
    text = path.read_text()
    
    # Use section-aware chunking
    #chunks = chunk_by_sections(text)
    chunks = chunk_by_sentences(text, sentences_per_chunk=2)
    
    # Add to ChromaDB
    collection.add(
        ids=[f"{path.stem}_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": str(path), "chunk_index": i} for i in range(len(chunks))]
    )
    
    return len(chunks)


def query(question: str, n_results: int = 5) -> list[dict]:
    """Retrieve relevant chunks."""
    results = collection.query(query_texts=[question], n_results=n_results)
    
    return [
        {"text": results["documents"][0][i], "source": results["metadatas"][0][i]["source"]}
        for i in range(len(results["documents"][0]))
    ]

def RAG_Tool(Query: Annotated[str, "The query to the RAG tool"]):
    #ingest("./Deep_Research_Reports/Deep_Research_Report.md")
    chunks = query(Query)

    results= []
    for c in chunks:
        results.append(c["text"])
    return results
    

if __name__ == "__main__":
    results = RAG_Tool("What methods are used to validate HPLC data quality?.")
    for r in results:
        print(r)
        print("\n" + "="*80 + "\n")