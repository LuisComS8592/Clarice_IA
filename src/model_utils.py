import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from sentence_transformers import SentenceTransformer
from functools import lru_cache


# Define o dispositivo globalmente
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)  # Cache para não recarregar os modelos da memória
def carregar_modelos_principais(modelo_path: str):
    """
    Carrega o model GPT-2 e o Tokenizer do disco.
    """
    print(f"Carregando model GPT-2 de '{modelo_path}' no {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(modelo_path)
    model = GPT2LMHeadModel.from_pretrained(modelo_path).to(DEVICE)
    model.eval()  # Coloca em modo de avaliação (desliga dropout, etc.)

    # Define o 'pad_token' se não existir, crucial para geração
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


@lru_cache(maxsize=1)
def carregar_modelo_embedder():
    """
    Carrega o model de embeddings (SBERT) para análise semântica.
    """
    print("Carregando model SBERT (Embeddings)...")
    # 'paraphrase-multilingual' é robusto para português
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


@lru_cache(maxsize=1)  # Reutiliza o pipeline carregado
def carregar_pipeline_geracao(modelo_path: str):
    """
    Carrega o pipeline 'text-generation' de forma otimizada para o app.
    """
    # DEVICE é importado do topo do seu model_utils.py
    # (DEVICE = "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Carregando pipeline de '{modelo_path}' no dispositivo: {DEVICE}...")
    device_id = 0 if DEVICE == "cuda" else -1

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(modelo_path)
        gerador = pipeline(
            'text-generation',
            model=modelo_path,
            tokenizer=tokenizer,
            device=device_id
        )

        # Garantia de Pad Token (boa prática)
        if gerador.tokenizer.pad_token is None:
            gerador.tokenizer.pad_token = gerador.tokenizer.eos_token
            gerador.model.config.pad_token_id = gerador.model.config.eos_token_id

        print("✅ Pipeline carregado com sucesso.")
        return gerador

    except Exception as e:
        print(f"❌ ERRO ao carregar pipeline: {e}")
        return None