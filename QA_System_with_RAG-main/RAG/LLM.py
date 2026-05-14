#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from typing import Dict, List, Optional, Tuple, Union

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""以下のコンテキストを使用して、ユーザーの質問に答えてください。答えがわからない場合は、「わかりません」と答えてください。常に日本語で回答してください。
        質問: {question}
        参考可能なコンテキスト：
        ···
        {context}
        ···
        与えられたコンテキストで回答できない場合は、「データベースにこの内容は存在しないため、わかりません」と回答してください。
        有用な回答:""",
    InternLM_PROMPT_TEMPALTE="""最初にコンテキストを要約し、その後コンテキストを使用してユーザーの質問に答えてください。答えがわからない場合は、「わかりません」と答えてください。常に日本語で回答してください。
        質問: {question}
        参考可能なコンテキスト：
        ···
        {context}
        ···
        与えられたコンテキストで回答できない場合は、「データベースにこの内容は存在しないため、わかりません」と回答してください。
        有用な回答:"""
)

STYLE_PROMPTS = {
    "概要レポート": """以下のコンテキストを使用して、ユーザーの質問に答えてください。答えがわからない場合は、「わかりません」と答えてください。常に日本語で回答してください。
        質問: {question}
        参考可能なコンテキスト：
        ···
        {context}
        ···
        
        【回答形式】概要レポート
        以下の要件に従って回答してください：
        - ソースから重要な分析インサイトと主要な引用を要約すること
        - 構造を明確にし、見出しを使ってポイントを整理すること
        - 箇条書きを活用し、素早く読める形式にすること
        - 各セクションの最後に簡潔な要約を添えること
        与えられたコンテキストで回答できない場合は、「データベースにこの内容は存在しないため、わかりません」と回答してください。
        有用な回答:""",

    "学習ガイド": """以下のコンテキストを使用して、ユーザーの質問に答えてください。答えがわからない場合は、「わかりません」と答えてください。常に日本語で回答してください。
        質問: {question}
        参考可能なコンテキスト：
        ···
        {context}
        ···
        
        【回答形式】学習ガイド
        以下の要素を必ず含めて、体系的な学習資料として回答してください：
        1. 簡答問題クイズ（3〜5問、各問に解答と解説を付けること）
        2. 推奨される論文の質問または探究テーマ（2〜3つ）
        3. 重要用語の用語集（5〜10語、各用語に簡潔な定義を付けること）
        与えられたコンテキストで回答できない場合は、「データベースにこの内容は存在しないため、わかりません」と回答してください。
        有用な回答:""",

    "ブログ記事": """以下のコンテキストを使用して、ユーザーの質問に答えてください。答えがわからない場合は、「わかりません」と答えてください。常に日本語で回答してください。
        質問: {question}
        参考可能なコンテキスト：
        ···
        {context}
        ···
        
        【回答形式】ブログ記事
        以下のスタイルで回答してください：
        - 重要なポイントを文章にまとめ、わかりやすく親しみやすい文体で書くこと
        - 専門用語は噛み砕いて説明し、一般読者にも理解できるようにすること
        - 導入・本文・まとめの構成で、読み物としての完成度を高めること
        - 適宜、比喩や具体例を用いて内容を身近に感じさせること
        与えられたコンテキストで回答できない場合は、「データベースにこの内容は存在しないため、わかりません」と回答してください。
        有用な回答:""",
}

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI
        client = OpenAI()   
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content

class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str='') -> str:
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response


    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        
        
class ZhipuAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = 'chatglm_lite') -> None:
        super().__init__(path)
        self.model = model
        self.api_key = self.get_api_key()

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        from zhipuai import ZhipuAI

        # Initialize the client with API Key
        client = ZhipuAI(api_key=self.api_key)
        
        # Format the prompt using the template
        full_prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)

        # Send the request
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            # return the response
            return response.choices[0].message.content
        except AttributeError as e:
            print(f"Attribute Error in ZhipuAI: {e}")
            return "ZHIPUAI API のレスポンス解析に失敗しました。APIを確認してください。"
        except Exception as e:
            print(f"ZhipuAI Error: {e}")
            return "ZHIPUAI API の呼び出しに失敗しました。設定を確認してください。"

    def chat_stream(self, prompt: str, history: List[dict], content: str, style: str = "カスタム形式", custom_instruction: str = "", max_tokens: int = 800):
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=self.api_key)

        if style in STYLE_PROMPTS:
            full_prompt = STYLE_PROMPTS[style].format(question=prompt, context=content)
        else:
            full_prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)
            if custom_instruction:
                full_prompt += f"\n\n【カスタム指示】\n{custom_instruction}"

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"ZhipuAI Stream Error: {e}")
            yield "ZHIPUAI API の呼び出しに失敗しました。設定を確認してください。"

    def get_api_key(self) -> str:
        import os
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY が設定されていません。環境変数に正しく設定されていることを確認してください")
        return api_key


class DeepSeekAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = 'deepseek-chat') -> None:
        super().__init__(path)
        self.model = model
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY が設定されていません。")

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

        full_prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek Error: {e}")
            return "DeepSeek API の呼び出しに失敗しました。"

    def chat_stream(self, prompt: str, history: List[dict], content: str, style: str = "カスタム形式", custom_instruction: str = "", max_tokens: int = 800):
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

        if style in STYLE_PROMPTS:
            full_prompt = STYLE_PROMPTS[style].format(question=prompt, context=content)
        else:
            full_prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)
            if custom_instruction:
                full_prompt += f"\n\n【カスタム指示】\n{custom_instruction}"

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"DeepSeek Stream Error: {e}")
            yield "DeepSeek API の呼び出しに失敗しました。"