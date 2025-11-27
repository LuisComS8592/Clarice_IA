# Clarice IA ✍️🤖

> *"Que ninguém se engane, só se consegue a simplicidade através de muito trabalho."*

Uma Inteligência Artificial Generativa baseada em **Small Language Models (SLMs)**, capaz de mimetizar o estilo literário complexo e introspectivo de **Clarice Lispector**.

Este projeto foi desenvolvido como um estudo de caso científico sobre os limites do *fine-tuning* em arquiteturas menores, culminando em uma análise adversarial rigorosa.

![Status](https://img.shields.io/badge/Status-Concluído-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Model](https://img.shields.io/badge/Model-GPT--2%20Small-purple)
![Tech](https://img.shields.io/badge/Tech-PyTorch%20%7C%20Streamlit-orange)

---

## 🧠 Sobre o Projeto

O objetivo deste trabalho foi investigar se um modelo pequeno (124M parâmetros), limitado em memória e abstração, conseguiria capturar a "alma" estilística de uma autora conhecida por sua sintaxe não-convencional e fluxo de consciência.

Utilizando um *corpus* curado de **4MB** contendo crônicas, contos e romances, aplicamos um protocolo de **"Overfitting Estilístico Controlado"** (remoção de regularização e *weight decay* zero).

### 📊 Resultados Principais

O modelo foi submetido a uma avaliação híbrida (Métrica + Qualitativa + Adversarial). Os resultados finais documentados no artigo foram:

| Métrica | Valor | Interpretação |
| :--- | :--- | :--- |
| **Perplexity (PPL)** | 26.14 | Alta fluidez gramatical em Português. |
| **Originalidade** | 99.29% | O modelo cria frases novas, sem plagiar o corpus. |
| **LLM-as-a-Judge** | **8.0/10** | Alta fidelidade estilística percebida (avaliado por Gemini). |
| **Teste Adversarial** | 85.83% | O classificador detectou a IA em 85% dos casos. |

### 🔍 Insights da Pesquisa
Embora a IA tenha recebido nota 8.0 pela qualidade estética, a análise adversarial revelou que ela opera criando uma **"Caricatura Existencialista"**:
* **Vícios da IA:** Repetição excessiva de temas centrais (*eu, silêncio, mundo, janela*).
* **O que faltou:** Variabilidade lexical e termos concretos (*agudez, acumulando, nomes próprios*) que ancoram a escrita da autora real.

---

## 🛠️ Instalação e Uso

Para rodar a Clarice IA localmente em sua máquina:

### 1. Clone o repositório
```bash
git clone [https://github.com/LuisComS8592/Clarice_IA.git](https://github.com/LuisComS8592/Clarice_IA.git)
cd Clarice_IA
```

### 2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

### 3. **Baixe o Modelo**
Devido ao tamanho dos pesos, o modelo não está hospedado diretamente no Git. Execute o script utilitário para baixá-lo automaticamente:
```
python download_model.py
```
Isso criará a pasta ``./model`` com os arquivos necessários.

### 4. **Execute a Interface:**
Inicie o aplicativo web local:
```bash
streamlit run app.py
```
O navegador abrirá automaticamente com o chat interativo.

## 📂 Estrutura
```
/Clarice_IA/
│
├── model/                           # Pasta onde o modelo será baixado
│
├── src/                             # Código Fonte Modular
│   ├── model_utils.py               # Carregamento otimizado do GPT-2
│   └── evaluation_metrics.py        # Cálculos de Perplexity, BERTScore e LLM Judge
│
├── app.py                           # Interface Web (Streamlit)
├── download_model.py                # Script de download automático do Drive
├── evaluate.py                      # Pipeline de auditoria e métricas
├── corpus_limpo.txt                 # Dataset curado e normalizado
├── artigo.pdf                       # Artigo científico completo com os resultados
├── requirements.txt                 # Dependências do projeto
├── training and evaluation.ipynb    # Notebook de treino e validação
└── GenAI_Classifier.ipynb           # Notebook da análise adversarial (O Duelo)
```
---
`Este projeto foi desenvolvido para fins acadêmicos e de portfólio na área de Processamento de Linguagem Natural (NLP) e IA Generativa.`
