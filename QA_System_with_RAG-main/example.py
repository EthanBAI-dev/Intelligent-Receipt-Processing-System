# 必要なモジュールとクラスをインポートします
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat, ZhipuAIChat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding


# ベクター作成プロセス
# ドキュメントを読み込んで分割します
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
# ベクターストアを初期化します
vector = VectorStore(docs)
# 埋め込みモデルを作成します
embedding = ZhipuEmbedding()
# 各ドキュメントをベクトル化します
vector.get_vector(EmbeddingModel=embedding)
# ベクトルとドキュメントをローカルストレージに保存します
vector.persist(path='storage')


# LLMコール
# ベクターストアを再初期化します
vector = VectorStore()
# ローカルに保存されたデータを読み込みます
vector.load_vector('./storage')
# 埋め込みモデルを再初期化します
embedding = ZhipuEmbedding()
# 質問内容を設定します
question = 'アジレント自動化ソリューションは？'
# ベクターストアを使って最も関連性の高い文書を取得します
content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
# print(content)
# LLM モデルを初期化します
chat = ZhipuAIChat(model='chatglm_lite')
# 質問に基づく回答を生成します
print(chat.chat(question, [], content))