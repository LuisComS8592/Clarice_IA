# # Clarice IA ✍️🤖

> "Que ninguém se engane, só se consegue a simplicidade através de muito trabalho."

Uma Inteligência Artificial Generativa fine-tuned no estilo literário de **Clarice Lispector**. Este projeto utiliza um modelo GPT-2 (Small) treinado em um corpus curado de crônicas e romances da autora, capaz de gerar textos introspectivos e filosoficamente densos.

![Status](https://img.shields.io/badge/Status-Concluído-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Tech](https://img.shields.io/badge/Tech-Transformers%20%7C%20Streamlit-orange)

## 🧠 Sobre o Projeto

O objetivo foi explorar os limites de modelos menores (GPT-2) na captura de estilos literários complexos. O modelo passou por:
1.  **Curadoria de Dados:** Limpeza de ruídos, OCR e formatação.
2.  **Fine-Tuning:** Treinamento com agendamento linear e *weight decay* zero para maximizar a apreensão estilística.
3.  **Avaliação:** Validado por métricas de *Perplexity*, *BERTScore* e avaliado qualitativamente pelo Gemini 2.5 Flash.

**Resultado:** O modelo atingiu uma nota **8.0/10** em avaliação qualitativa, demonstrando alta originalidade (>99%) e captura da sintaxe peculiar da autora.

## 🛠️ Instalação e Uso

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/LuisComS8592/Clarice_IA.git](https://github.com/LuisComS8592/Clarice_IA.git)
   cd Clarice_IA
   ```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Obtenha o Modelo:**
- Execute o script ``download_model.py`` para baixar os pesos do modelo.
- Você pode treinar o seu próprio usando o script de notebook fornecido.

4. **Execute a Interface:**
```bash
streamlit run app.py
```

## 📂 Estrutura
```
/Clarice_IA/
│
├── model/                           # Pasta destino do modelo
│
├── src/                             # "Source": Onde vive nossa lógica principal
│   ├── model_utils.py               # Funções para carregar modelos (GPT-2, SBERT)
│   └── evaluation_metrics.py        # Todas as nossas funções de métrica (PPL, Juiz LLM, etc)
│
├── app.py                           # App Streamlit (Interface)
├── evaluate.py                      # O SCRIPT que você executa para rodar a avaliação
├── corpus.txt                       # O corpus de dados
├── download_model.py                # O corpus de dados
├── requirements.txt                 # Nossas dependências
└── training and evaluation.ipynb    # Notebook utilizado para treinamento e avaliação do modelo no Colab
```

---
`Desenvolvido como projeto de portfólio de Data Science.`
