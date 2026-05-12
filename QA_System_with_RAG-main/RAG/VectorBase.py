#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
模块名称: 轻量级本地向量数据库 (Vector Store)
功能描述:
    本模块实现了一个用于 RAG (检索增强生成) 系统的基础向量数据库。
    它充当了 LLM 的“外部记忆库”，主要负责以下核心工作流：

    1. 向量化 (Embedding): 接收切分好的文本块，调用指定的 Embedding 模型将其转化为高维向量。
    2. 持久化 (Persistence): 将计算好的高维向量和原始文本以 JSON 格式保存在本地硬盘，避免重复计算的开销。
    3. 相似度检索 (Query): 当用户输入提问时，将提问也转化为向量，并通过计算“余弦相似度 (Cosine Similarity)”
       与库中的所有向量进行比对，最终召回并返回最匹配的 Top-K 个文本块。

依赖说明:
    - 依赖 `RAG.Embeddings` 中的基础模型接口。
    - 使用 `numpy` 进行高效的矩阵运算和排序。
    - 使用 `tqdm` 提供长文本处理时的可视化进度条。
=============================================================================
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import json
from RAG.Embeddings import BaseEmbeddings, OpenAIEmbedding, JinaEmbedding, ZhipuEmbedding
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        """
        初始化向量数据库。
        :param document: 待处理的文档列表（字符串数组）。
        """
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        """
        遍历所有文档，调用传入的嵌入模型生成向量。
        """
        self.vectors = []
        # 使用 tqdm 包装循环，在终端显示生成向量的进度条 doc已经是段落了 tqdm是一个进度条显示工具，desc是说明文字
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))# 把文本进行向量化
        return self.vectors

    def persist(self, path: str = 'storage'):
        """
        将文档数据和对应的向量数据持久化保存到本地 JSON 文件中。
        """
        # 如果指定的保存目录不存在，则创建它
        if not os.path.exists(path):
            os.makedirs(path)

        # 1. 保存原始文档内容
        with open(f"{path}/documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)

        # 2. 如果向量已经生成，则将其保存为 vectors.json
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        vectors_path = f"{path}/vectors.json"
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(
                f"ベクトルファイルが見つかりません: {vectors_path}\n"
                f"最初にデータをベクトル化して保存してください。"
            )

        with open(vectors_path, 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)

        doc_path_new = f"{path}/documents.json"
        doc_path_old = f"{path}/doecment.json"
        if os.path.exists(doc_path_new):
            doc_path = doc_path_new
        elif os.path.exists(doc_path_old):
            doc_path = doc_path_old
        else:
            raise FileNotFoundError(
                f"文書ファイルが見つかりません: {doc_path_new}\n"
                f"最初にデータをベクトル化して保存してください。"
            )

        with open(doc_path, 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度。值越接近 1，表示两段文本语义越相关。
        """
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        """
        检索与用户提问最相关的文档。
        :param query: 用户输入的提问字符串。
        :param EmbeddingModel: 用于将提问转化为向量的嵌入模型。
        :param k: 需要返回的最相关文档的数量 (Top-K)。
        """
        # 1. 将用户的提问文本转化为向量
        query_vector = EmbeddingModel.get_embedding(query)

        # 2. 遍历库中所有向量，计算提问向量与库中每个向量的相似度得分
        result = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])

        # 3. 对相似度得分进行排序，提取得分最高的倒数前 k 个索引 (argsort 是从小到大排)
        #    [::-1] 用于将结果反转，使其按照从大到小（最相关到次相关）的顺序排列
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()