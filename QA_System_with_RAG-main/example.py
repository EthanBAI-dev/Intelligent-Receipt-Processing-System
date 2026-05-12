# 必要なモジュールとクラスをインポートします
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat, ZhipuAIChat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding


# 向量创建过程
# 读取并分割文档
#   - max_token_len=600: 每个文本块的最大长度限制为 600 个 Token (模型处理文本的基础单位)。
#   - cover_content=150: 文本块之间的重叠长度为 150 个 Token。这被称为“重叠区 (Overlap)”，目的是防止一句话或一个段落被硬生生切断，从而保留上下文的连贯性。
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
# 初始化向量库 (Vector Store)
vector = VectorStore(docs)
# 创建嵌入模型 (Embedding Model)
embedding = ZhipuEmbedding()
# 4. 执行文本向量化 (核心计算阶段)
# 调用向量库的 get_vector 方法，并指定使用刚才创建的智谱 Embedding 模型。
# 这一步会遍历所有的文档块，通过调用 API 将它们逐一转换为向量，并将这些向量与原文配对缓存在内存中。
vector.get_vector(EmbeddingModel=embedding)
# 将向量和文档保存到本地存储
vector.persist(path='storage')


# 调用 LLM
# 重新初始化向量库
vector = VectorStore()
# 读取本地保存的数据
vector.load_vector('./storage')
# 重新初始化嵌入模型
embedding = ZhipuEmbedding()
# 设置提问内容
question = '经济本质是什么？'
# 使用向量库检索出相关性最高的文档
#   - k=1: 这是一个关键参数，表示“只返回最相关的一个文本块”。
#     如果设为 k=3，则会返回相似度最高的前三个片段。
# [0]: 因为 query 方法返回的是一个列表（List），[0] 表示取列表中最匹配的那第一个片段。
content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
# print(content)
# 初始化 LLM 模型
chat = ZhipuAIChat(model='chatglm_lite')
# 基于检索到的内容和提问生成回答
#   - question: 原始提问。
#   - []: 这是一个占位符，通常用于存放“对话历史 (History)”。
#     如果你想实现多轮对话（即模型记得你上一句说了什么），需要把之前的对话记录传进去。
#   - content: 这是刚才从向量库里检索出来的“参考资料”。
#     大模型会收到类似于这样的提示词：“请参考以下内容：{content}，回答问题：{question}”
print(chat.chat(question, [], content))