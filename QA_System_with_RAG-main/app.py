import os
import re
import html as html_mod
import streamlit as st
from RAG.VectorBase import VectorStore
from RAG.Embeddings import ZhipuEmbedding
from RAG.LLM import ZhipuAIChat, DeepSeekAIChat

st.set_page_config(page_title="RAGシステム", layout="wide")
st.title("RAGインテリジェントQ&Aシステム")

MODEL_OPTIONS = {
    "DeepSeek-V4": "deepseek-chat",
    "DeepSeek-Flash": "deepseek-chat",
}

MODEL_PROVIDER = {
    "DeepSeek-V4": "deepseek",
    "DeepSeek-Flash": "deepseek",
}

ANSWER_STYLES = {
    "概要レポート": "概要レポート",
    "学習ガイド": "学習ガイド",
    "ブログ記事": "ブログ記事",
    "カスタム形式": "カスタム形式",
}

LENGTH_OPTIONS = [200, 400, 800]


def strip_html(raw: str) -> str:
    text = html_mod.unescape(raw)
    text = re.sub(r'<[^>]*>', '', text, flags=re.DOTALL)
    return text


def highlight_text(text, query):
    if not query:
        return text
        
    # 1. 基础转义，防止破坏 HTML
    clean_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # 2. 提取英文/数字组合（如 RAG, DeepSeek, 100K）
    eng_num = re.findall(r'[a-zA-Z0-9]{2,}', query)
    
    # 3. 提取中/日文进行“滑动窗口”切词
    # 先剔除标点符号和英文
    cjk_pure = re.sub(r'[^\u4e00-\u9fa5\u3040-\u30ff]', '', query)
    
    # 过滤掉常见的无意义停用词
    stopwords = ["的", "是", "在", "了", "和", "与", "有", "就", "不", "都", "一", "什么", "怎么", "如何"]
    for sw in stopwords:
        cjk_pure = cjk_pure.replace(sw, "")
        
    cjk_words = []
    if len(cjk_pure) >= 2:
        # 生成 4字、3字、2字 的词组组合
        for length in [4, 3, 2]:
            for i in range(len(cjk_pure) - length + 1):
                cjk_words.append(cjk_pure[i:i+length])
    elif len(cjk_pure) == 1:
        cjk_words.append(cjk_pure)
        
    # 4. 合并关键词并去重
    keywords = list(set(eng_num + cjk_words))
    
    if not keywords:
        return clean_text
        
    # 按长度降序排列，确保优先匹配长词
    keywords.sort(key=len, reverse=True)
    
    # 5. 安全的高亮替换
    pattern = re.compile(f"({'|'.join(map(re.escape, keywords))})", flags=re.IGNORECASE)
    parts = pattern.split(clean_text)
    
    result = ''
    for part in parts:
        if part and pattern.fullmatch(part):
            # 恢复为你原先的干净样式：无背景，只有红色加粗字体
            result += f'<span style="color:#ff4b4b;font-weight:bold">{part}</span>'
        else:
            result += part
            
    return result


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
        docs, sources = ReadFiles(data_dir).get_content(max_token_len=600, cover_content=150)
        vector = VectorStore(docs, sources)
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

if "answer_style" not in st.session_state:
    st.session_state.answer_style = "概要レポート"

if "answer_length" not in st.session_state:
    st.session_state.answer_length = 400

if "custom_instruction" not in st.session_state:
    st.session_state.custom_instruction = ""

with st.sidebar:
    st.header("設定")

    st.selectbox(
        "LLMモデル",
        options=list(MODEL_OPTIONS.keys()),
        key="model_name"
    )

    k = st.slider("検索数 Top-K", 1, 5, 3)

    show_context = st.checkbox("検索内容を表示", True)

    if st.button("会話をクリア"):
        st.session_state.history = []
        st.session_state.context_docs = []
        st.session_state.answer_style = "概要レポート"
        st.session_state.answer_length = 200
        st.session_state.custom_instruction = ""
        st.rerun()

    st.divider()
    st.subheader("回答設定")

    st.selectbox(
        "回答の長さ（トークン数）",
        options=LENGTH_OPTIONS,
        key="answer_length",
        format_func=lambda x: f"{x} トークン",
    )

    st.selectbox(
        "回答スタイル",
        options=list(ANSWER_STYLES.keys()),
        key="answer_style",
    )

    if st.session_state.answer_style == "カスタム形式":
        st.text_area(
            "カスタム指示",
            key="custom_instruction",
            placeholder="例：箇条書きで簡潔に答えてください。",
        )

    st.divider()
    st.header("ナレッジベース")

    data_dir = './data'
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(('.md', '.txt', '.pdf'))]
        if files:
            for f in sorted(files):
                st.markdown(f"{f}")
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
                context = "\n".join([d["text"] for d in docs])

                style = st.session_state.answer_style
                custom_instruction = ""
                if style == "カスタム形式" and st.session_state.get("custom_instruction", "").strip():
                    custom_instruction = st.session_state.custom_instruction
                max_tokens = st.session_state.answer_length

                chat = get_chat_model(st.session_state.model_name)
                answer = st.write_stream(chat.chat_stream(
                    question, [], context,
                    style=style,
                    custom_instruction=custom_instruction,
                    max_tokens=max_tokens,
                ))

            except Exception as e:
                answer = f"システムエラーが発生しました。\n\nエラー詳細：{e}"
                st.write(answer)

        st.session_state.history.append(("assistant", answer))

if show_context and st.session_state.context_docs:
    with st.expander("検索内容（Top-K）", expanded=False):
        # 获取用户的提问
        last_question = ""
        if len(st.session_state.history) >= 2:
            last_question = st.session_state.history[-2][1]
        elif len(st.session_state.history) == 1:
            last_question = st.session_state.history[0][1]

        for i, doc in enumerate(st.session_state.context_docs):
            text = doc["text"]
            score = doc.get("score", 0)
            source = doc.get("source", "")

            text_clean = strip_html(text)

            st.markdown(f"### Chunk {i+1}")
            st.caption(f"出典: {source}  |  類似度: {score:.4f}")

            highlighted = highlight_text(text_clean, last_question)
            
            # 注意这里我删掉了 f-string 内部的换行符 \n，防止 Streamlit 继续触发 Markdown 代码块渲染
            html_content = (
                f'<div style="border:1px solid #ddd; padding:15px; border-radius:10px; margin-bottom:15px; line-height:1.8;">'
                f'{highlighted}'
                f'</div>'
            )
            
            st.markdown(html_content, unsafe_allow_html=True)
