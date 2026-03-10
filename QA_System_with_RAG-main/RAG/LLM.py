#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
            max_tokens=150,
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
                max_tokens=150
            )
            # return the response
            return response.choices[0].message.content
        except AttributeError as e:
            print(f"Attribute Error in ZhipuAI: {e}")
            return "ZHIPUAI API のレスポンス解析に失敗しました。APIを確認してください。"
        except Exception as e:
            print(f"ZhipuAI Error: {e}")
            return "ZHIPUAI API の呼び出しに失敗しました。設定を確認してください。"

    def get_api_key(self) -> str:
        import os
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY が設定されていません。環境変数に正しく設定されていることを確認してください")
        return api_key