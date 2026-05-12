import os
import re
import streamlit as st
from RAG.VectorBase import VectorStore
from RAG.Embeddings import ZhipuEmbedding
from RAG.LLM import ZhipuAIChat, DeepSeekAIChat

st.set_page_config(page_title="RAGシステム", layout="wide")
st.title("📚 RAGインテリジェントQ&Aシステム")

MODEL_OPTIONS = {
    "DeepSeek-V4": "deepseek-chat",
    "DeepSeek-Flash": "deepseek-chat",
}

MODEL_PROVIDER = {
    "DeepSeek-V4": "deepseek",
    "DeepSeek-Flash": "deepseek",
}


def highlight_text(text, query):
    if not query:
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    escaped = re.escape(query)
    return re.sub(
        f'({escaped})',
        r'<span style="color:#ff4b4b;font-weight:bold">\1</span>',
        text,
        flags=re.IGNORECASE
    )


def _need_rebuild(data_dir, vectors_path):
    if not os.path.exists(vectors_path):
        return True
    vector_mtime = os.path.getmtime(vectors_path)
    if not os.path.isdir(data_dir):
        return False
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.endswith(('.md', '.txt', '.pdf'))]
    if not data_files:
        return False
    latest_data_mtime = max(os.path.getmtime(f) for f in data_files)
    return latest_data_mtime > vector_mtime


@st.cache_resource
def load_knowledge_base():
    from RAG.utils import ReadFiles

    data_dir = './data'
    storage_dir = './storage'
    vectors_path = f'{storage_dir}/vectors.json'
    embedding = ZhipuEmbedding()

    if _need_rebuild(data_dir, vectors_path):
        print('🔄 新しい文書を検出しました。ベクトルを自動再構築中...')
        docs = ReadFiles(data_dir).get_content(max_token_len=600, cover_content=150)
        vector = VectorStore(docs)
        vector.get_vector(EmbeddingModel=embedding)
        vector.persist(path=storage_dir)
        print(f'✅ ベクトル再構築が完了しました！合計 {len(docs)} チャンク')
    else:
        vector = VectorStore()
        vector.load_vector(storage_dir)

    return vector, embedding


def get_chat_model(model_name: str):
    provider = MODEL_PROVIDER.get(model_name, "deepseek")
    model_id = MODEL_OPTIONS.get(model_name, "deepseek-chat")

    if provider == "deepseek":
        return DeepSeekAIChat(model=model_id)

    return DeepSeekAIChat(model=model_id)


vector, embedding = load_knowledge_base()

if "history" not in st.session_state:
    st.session_state.history = []

if "context_docs" not in st.session_state:
    st.session_state.context_docs = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "DeepSeek-V4"

with st.sidebar:
    st.header("⚙️ 設定")

    st.selectbox(
        "🤖 LLMモデル",
        options=list(MODEL_OPTIONS.keys()),
        key="model_name"
    )

    k = st.slider("検索数 Top-K", 1, 5, 3)

    show_context = st.checkbox("検索内容を表示", True)

    if st.button("🧹 会話をクリア"):
        st.session_state.history = []
        st.session_state.context_docs = []
        st.rerun()

    st.divider()
    st.header("📁 ナレッジベース")

    data_dir = './data'
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(('.md', '.txt', '.pdf'))]
        if files:
            for f in sorted(files):
                st.markdown(f"📄 {f}")
        else:
            st.caption("資料がありません")
    else:
        st.caption("資料ディレクトリが存在しません")

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

question = st.chat_input("質問を入力してください...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            try:
                docs = vector.query(question, EmbeddingModel=embedding, k=k)
                st.session_state.context_docs = docs
                context = "\n".join(docs)

                chat = get_chat_model(st.session_state.model_name)
                answer = st.write_stream(chat.chat_stream(question, [], context))

            except Exception as e:
                answer = f"❌ システムエラーが発生しました。\n\nエラー詳細：{e}"
                st.write(answer)

        st.session_state.history.append(("assistant", answer))

if show_context and st.session_state.context_docs:
    with st.expander("📖 検索内容（Top-K）", expanded=False):
        last_question = st.session_state.history[-1][1] if st.session_state.history else ""
        for i, doc in enumerate(st.session_state.context_docs):
            st.markdown(f"### 📄 チャンク {i+1}")
            highlighted = highlight_text(doc, last_question if last_question else "")
            st.markdown(
                f"""
                <div style="
                    border:1px solid #444;
                    padding:15px;
                    border-radius:10px;
                    margin-bottom:10px;
                ">
                {highlighted}
                </div>
                """,
                unsafe_allow_html=True
            )
