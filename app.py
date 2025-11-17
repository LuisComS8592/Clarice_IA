import streamlit as st
import time

# Importa nossa função de carregamento do módulo 'src'
from src.model_utils import carregar_pipeline_geracao, DEVICE

# --- Constantes ---
CAMINHO_MODELO = "./model"

# --- Configuração da Página ---
st.set_page_config(page_title="Clarice.IA", page_icon="✍️", layout="centered")

# --- CSS ---
st.markdown("""
<style>
    /* 1. Definição das Variáveis de Tema */
    :root {
        --font-serif: 'Georgia', 'Times New Roman', serif;
        --cor-acento: #8B4513;      /* Sépia / Tinta de Caneta */

        /* MODO CLARO (Default) */
        --cor-fundo-claro: #f7f3e9; /* Papel antigo */
        --cor-texto-claro: #2a2a2a; /* Tinta preta */
        --cor-balao-user-claro: #FFFFFF;
        --cor-borda-clara: #e0e0e0;

        /* MODO ESCURO (Ativado via @media) */
        --cor-fundo-escuro: #1E1E1E; /* Fundo carvão */
        --cor-texto-escuro: #E0E0E0; /* Texto giz */
        --cor-balao-user-escuro: #2C2C2C; /* Balão cinza escuro */
        --cor-borda-escura: #444444;
    }

    /* 2. Aplica o Tema Padrão (Modo Claro) */
    .stApp {
        background-color: var(--cor-fundo-claro);
    }

    body, .stMarkdown, .stButton > button {
        color: var(--cor-texto-claro);
        font-family: var(--font-serif);
    }

    h1 {
        color: var(--cor-acento);
        font-family: var(--font-serif);
        font-weight: 300;
        text-align: center;
    }

    /* Balão da IA (Sempre usa o acento) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent-assistant"]) {
        background-color: transparent;
        border-left: 3px solid var(--cor-acento);
        padding-left: 1.5rem;
    }

    /* Balão do Usuário (Claro) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent-user"]) {
        background-color: var(--cor-balao-user-claro);
        border: 1px solid var(--cor-borda-clara);
        border-radius: 10px;
    }

    /* Input (Claro) */
    [data-testid="stChatInput"] {
        background-color: var(--cor-balao-user-claro);
        border-top: 1px solid var(--cor-borda-clara);
    }

    /* 3. Aplica Overrides para MODO ESCURO */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: var(--cor-fundo-escuro);
        }

        body, .stMarkdown, .stButton > button {
            color: var(--cor-texto-escuro);
        }

        /* Balão do Usuário (Escuro) */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent-user"]) {
            background-color: var(--cor-balao-user-escuro);
            border: 1px solid var(--cor-borda-escura);
        }

        /* Input (Escuro) */
        [data-testid="stChatInput"] {
            background-color: var(--cor-fundo-escuro);
            border-top: 1px solid var(--cor-borda-escura);
        }
    }

    /* Esconde menu e rodapé */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- Textos da Interface ---
st.title("Clarice.IA")
st.markdown(
    "<div style='text-align: center; color: #666; margin-bottom: 30px;'><i>“Escreva o início de um pensamento. Ela continuará.”</i></div>",
    unsafe_allow_html=True)


# --- Carregamento do Modelo ---
@st.cache_resource
def carregar_modelo_cacheado():
    """Função wrapper para o cache do Streamlit."""
    return carregar_pipeline_geracao(CAMINHO_MODELO)


gerador = carregar_modelo_cacheado()

if gerador is None:
    st.error(f"❌ Erro: Não foi possível carregar o modelo da pasta '{CAMINHO_MODELO}'.")
    st.stop()

# --- Lógica do Chat (sem alterações) ---
if "msgs" not in st.session_state:
    st.session_state.msgs = []

for msg in st.session_state.msgs:
    avatar = "👤" if msg["role"] == "user" else "🖋️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

prompt = st.chat_input("Escreva o início de um pensamento...")

if prompt:
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🖋️"):
        placeholder = st.empty()
        with st.spinner("Pensando..."):
            try:
                res = gerador(
                    prompt,
                    num_return_sequences=1,
                    temperature=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )

                full_text = res[0]['generated_text']

                if full_text.startswith(prompt):
                    clean_text = full_text[len(prompt):].strip()
                else:
                    clean_text = full_text.strip()

                if clean_text and clean_text[0] in [".", ",", "!", "?"]:
                    clean_text = clean_text[1:].strip()

                placeholder.write(clean_text)
                st.session_state.msgs.append({"role": "assistant", "content": clean_text})

            except Exception as e:
                st.error(f"Erro na geração do texto: {e}")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    if st.button("Limpar Conversa"):
        st.session_state.msgs = []
        st.rerun()