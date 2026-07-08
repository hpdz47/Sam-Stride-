FROM vllm/vllm-openai:nightly

RUN apt-get update && \
    apt-get install -y git 

RUN pip install --no-cache-dir --upgrade \
    uv \
    matplotlib \
    python-dotenv \
    "ag2[openai,browser-use,rag,lmm]==0.10.5"
    
RUN pip install --no-cache-dir --upgrade \
    protobuf==6.33.5

# Spectral analysis stack for the host-guest NMR/MS decision tools
# (Tools/Spectral_Analysis.py, Tools/Spectral_Ingestion.py).
RUN pip install --no-cache-dir --upgrade \
    numpy \
    scipy \
    nmrglue \
    dtw-python

RUN apt-get update && \
    apt-get install -y texlive-latex-extra 

RUN python3 -m playwright install --with-deps chromium