# ---------------------Imports -----------------------------
import sys
import os
import socket

from vLLM_Configuration import VLLM_Config
from dotenv import load_dotenv
from pathlib import Path

from autogen.agents.experimental import DocAgent
from autogen.agents.experimental.document_agent.chroma_query_engine import VectorChromaQueryEngine

# For local embeddings (optional - default uses all-MiniLM-L6-v2)
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# For local LLM with LlamaIndex (vLLM is OpenAI-compatible)
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

if __name__ == "__main__":
    load_dotenv('/RAG/.env')

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Hostname: {hostname} | IP Address: {ip_address}")
    os.environ['PATH'] = '/pip/bin:' + os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')

    # AG2 LLM config for DocAgent (the agent orchestration)
    llm_config = VLLM_Config(
        api_type="openai",
        cache_seed=None,
        temperature=0.0,
        enable_thinking=False,
        LLM_Type="Reasoning",
        IP_Address=ip_address
    ).build_config()

    # LlamaIndex LLM for the query engine (vLLM is OpenAI-compatible)
    # This is what processes RAG queries - DIFFERENT from AG2's LLMConfig
    llamaindex_llm = OpenAILike(
        model=os.environ["LLM_Model_Reasoning"],  # Match your vLLM model
        api_base=f"http://{ip_address}:8000/v1",  # Your vLLM server URL
        api_key=os.environ["API_KEY"],  # Your vLLM API key
        temperature=0.3,
        is_chat_model=True,
    )

    # Local embedding function (optional - default is all-MiniLM-L6-v2)
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # Create the query engine with correct parameters
    qe = VectorChromaQueryEngine(
        collection_name="Docs",          # Collection name
        db_path="/DB",              # Where to persist the database
        embedding_function=embedder,     # Local embeddings
        llm=llamaindex_llm,              # Local LLM for query processing
    )


    # Create DocAgent with both configs
    agent = DocAgent(
        query_engine=qe,
        llm_config=llm_config,
        collection_name="Docs",
        parsed_docs_path="/temp"
    )

    

    # Run query
    result = agent.run(message=sys.argv[1])
    result.process()
    
    # Print the answer (for stdout capture)
    if result.chat_history:
        print(result.chat_history[-1]["content"])
    else:
        print("No response generated")