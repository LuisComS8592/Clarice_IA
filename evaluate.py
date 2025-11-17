import pandas as pd
import random
import warnings
from typing import List

# Importa nossas funções locais dos módulos 'src'
from src.model_utils import carregar_modelos_principais, carregar_modelo_embedder
from src.evaluation_metrics import (
    calc_perplexity,
    calc_novelty,
    calc_embeddings_distance,
    calc_bertscore,
    llm_judge
)

# --- CONFIGURAÇÕES GLOBAIS ---
CAMINHO_MODELO = "./model"
ARQUIVO_CORPUS = "corpus.txt"
GEMINI_API_KEY = "COLE_SUA_CHAVE_API_AQUI"
N_AMOSTRAS = 15  # Quantidade de textos para gerar e avaliar


def gerar_amostras(model, tokenizer, n_amostras: int) -> List[str]:
    """Gera textos da IA para serem avaliados."""
    print(f"✍️ Gerando {n_amostras} amostras para avaliação...")
    prompts = ["O silêncio", "A vida", "Eu não", "O amor", "De repente", "A janela", "Senti que"]
    textos_gerados = []

    for _ in range(n_amostras):
        prompt = random.choice(prompts)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=150,
                do_sample=True,
                temperature=0.9,  # O "Sweet spot" que encontramos
                repetition_penalty=1.1,  # Penalidade leve
                pad_token_id=tokenizer.eos_token_id
            )
        texto = tokenizer.decode(out[0], skip_special_tokens=True)
        textos_gerados.append(texto)

    return textos_gerados


def carregar_corpus_real(caminho: str) -> (str, List[str]):
    """Carrega o texto bruto e uma lista de parágrafos de referência."""
    with open(caminho, "r", encoding="utf-8") as f:
        raw_corpus = f.read()
    real_docs = [p for p in raw_corpus.split('\n\n') if len(p) > 50]
    return raw_corpus, real_docs


def main():
    """Função principal que orquestra a avaliação."""
    warnings.filterwarnings("ignore")
    print("🚀 INICIANDO BATERIA DE TESTES DE AVALIAÇÃO...")

    # 1. Carregar Modelos e Dados
    model, tokenizer = carregar_modelos_principais(CAMINHO_MODELO)
    embedder = carregar_modelo_embedder()
    raw_corpus, real_docs = carregar_corpus_real(ARQUIVO_CORPUS)

    print(f"✅ Corpus carregado: {len(real_docs)} parágrafos de referência.")

    # 2. Geração
    textos_gerados = gerar_amostras(model, tokenizer, N_AMOSTRAS)

    # 3. Cálculo das Métricas
    print("📐 Calculando métricas...")
    # Pega amostras aleatórias do real para comparar com os gerados
    referencias = random.sample(real_docs, min(N_AMOSTRAS, len(real_docs)))

    ppl = calc_perplexity(model, tokenizer, "\n\n".join(referencias))
    novelty = calc_novelty(textos_gerados, raw_corpus, n=5)
    bert_f1 = calc_bertscore(textos_gerados, referencias)
    emb_sim = calc_embeddings_distance(textos_gerados, referencias, embedder)
    nota_llm = llm_judge(textos_gerados[0], GEMINI_API_KEY)  # Avalia só o primeiro

    # 4. Montagem do Relatório
    dados = {
        "Métrica": [
            "Perplexity (Incerteza)",
            "Originalidade (Novelty)",
            "BERTScore (F1)",
            "Similaridade Estilo (Embeddings)",
            "LLM Judge"
        ],
        "Valor": [
            f"{ppl:.2f}",
            f"{novelty:.2%}",
            f"{bert_f1:.4f}",
            f"{emb_sim:.4f}",
            nota_llm
        ],
        "Interpretação Ideal": [
            "Baixo (< 30)",
            "Alto (> 90%)",
            "Alto (> 0.7)",
            "Alto (> 0.6)",
            "Alto (> 8.0)"
        ]
    }

    df = pd.DataFrame(dados)

    print("\n\n============== RELATÓRIO FINAL ==============")
    print(f"Amostra gerada: '{textos_gerados[0][:100]}...'\n")
    print(df.to_string())  # .to_string() imprime bonito no terminal


# Padrão de execução profissional
if __name__ == "__main__":
    main()