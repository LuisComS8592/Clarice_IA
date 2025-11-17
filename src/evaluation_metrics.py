import torch
import numpy as np
import google.generativeai as genai
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_calc
from nltk.util import ngrams
from typing import List, Dict, Any

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calc_perplexity(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, texto: str) -> float:
    """Métrica de Incerteza (Menor é melhor)"""
    encodings = tokenizer(texto, return_tensors="pt")
    max_len = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_len, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
        prev_end_loc = end_loc
        if end_loc == seq_len: break

    return torch.exp(torch.stack(nlls).mean()).item()


def calc_novelty(gerados: List[str], corpus_raw: str, n: int = 5) -> float:
    """
    Mede originalidade (N-Grams).
    1.0 = Totalmente novo. 0.0 = Cópia total.
    """
    tokens_corpus = corpus_raw.split()
    corpus_ngrams = set(ngrams(tokens_corpus, n))
    scores = []

    for texto in gerados:
        tokens_gen = texto.split()
        if len(tokens_gen) < n: continue
        gen_ngrams = list(ngrams(tokens_gen, n))

        novos = sum(1 for ng in gen_ngrams if ng not in corpus_ngrams)
        ratio = novos / len(gen_ngrams) if len(gen_ngrams) > 0 else 0
        scores.append(ratio)

    return np.mean(scores)


def calc_embeddings_distance(gerados: List[str], refs: List[str], embedder: SentenceTransformer) -> float:
    """Calcula a Distância Cosseno entre gerados e reais."""
    emb_real = embedder.encode(refs, convert_to_tensor=True)
    emb_gen = embedder.encode(gerados, convert_to_tensor=True)

    cosine_scores = util.cos_sim(emb_gen, emb_real)
    return torch.mean(cosine_scores).item()


def calc_bertscore(gerados: List[str], refs: List[str]) -> float:
    """Calcula o F1 do BERTScore."""
    P, R, F1 = bert_score_calc(gerados, refs, lang="pt", verbose=False, device=DEVICE)
    return F1.mean().item()


def llm_judge(texto_para_avaliar: str, api_key: str) -> str:
    """Usa o Gemini para dar uma nota qualitativa."""
    if not api_key:
        return "N/A (Sem chave API)"

    genai.configure(api_key=api_key)
    try:
        model_judge = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return f"Erro ao carregar Juiz: {e}"

    prompt = f"""
    Atue como um crítico literário especialista em Clarice Lispector.
    Avalie o texto abaixo gerado por uma IA.

    Texto: "{texto_para_avaliar}"

    Seja rigoroso. A Clarice usa pontuação estranha, paradoxos, fluxo de consciência denso.
    Se o texto for genérico ou parecer um diário comum, dê nota baixa.

    Responda APENAS com uma nota de 0 a 10.
    """

    try:
        response = model_judge.generate_content(prompt)
        return response.text.strip().replace('\n', ' | ')
    except Exception as e:
        return f"Erro API: {e}"