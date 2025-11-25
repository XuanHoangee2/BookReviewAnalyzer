from typing import Optional
from transformers import AutoTokenizer
import re
from datasets import Dataset
import os
from huggingface_hub import login

hf_token = os.getenv('HF_TOKEN')
login(hf_token,add_to_git_credential=True)

#Constant
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
MIN_TOKEN = 5
MAX_TOKEN = 90
MIN_CHARS = 15
CEILING_CHARS = MAX_TOKEN* 7

class BookItems:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code = True)
    QUESTION = "Analyze the following product review and provide both the rating and a short summary."
    PREFIX = "\n\nOutput:\n"
    prompt: Optional[str] = None
    token_count: int = 0
    title: str
    rating: float
    text: str
    verified_purchase: bool

    def __init__(self,data):
        self.data = data
        self.title = data["title"]
        self.text = data["text"]
        self.rating = data["rating"]
    def clean_text(self,stuff):
        if not stuff:
            return ""
        if not isinstance(stuff,str):
            stuff = str(stuff)
        #Remove unnecessary characters
        stuff = re.sub(r'[:\[\]"{}\s]+',' ',stuff).strip()
        stuff = stuff.replace(" ,",",").replace(",,,",",").replace(",,",",")
        stuff = stuff.split(' ')
        return " ".join(stuff)
    
    def parse(self):
        self.text = self.clean_text(self.text)
        if self.tokenizer:
            num_tokens = len(self.tokenizer.encode(self.text, add_special_tokens=False))
        else:
            num_tokens = len(self.text.split())

        if len(self.text) < MIN_CHARS or num_tokens < MIN_TOKEN:
            return None 
        if num_tokens > MAX_TOKEN:
            if self.tokenizer:
                truncated = self.tokenizer.encode(self.text, add_special_tokens=False)[:MAX_TOKEN]
                self.text = self.tokenizer.decode(truncated)
            else:
                self.text = " ".join(self.text.split()[:MAX_TOKEN])
            return self.text

    def MakePrompt(self):
        self.text = self.parse()
        if self.text is None:
            return None
        prompt = f"{self.QUESTION}\n\nReview:\n{self.text}\n{self.PREFIX}"
        if self.rating is not None and self.title is not None:
            prompt += f"Rating: {str(self.rating)}\nSummary: {self.title}"
        else:
            prompt += "Rating: [missing]\nSummary: [missing]"
        self.prompt = prompt

    
    def test_prompt(self):
        return f"{self.QUESTION}\n\nReview:\n{self.text}\n{self.PREFIX}"
    
    def __repr__(self):
        """
        String representation for debugging.
        """
        return f"<ReviewItem ={self.rating}, summary={self.title[:30] if self.title else None}>"


# def clean_text(stuff):
#         if not stuff:
#             return ""
#         if not isinstance(stuff,str):
#             stuff = str(stuff)
#         #Remove unnecessary characters
#         stuff = re.sub(r'[:\[\]"{}\s]+',' ',stuff).strip()
#         stuff = stuff.replace(" ,",",").replace(",,,",",").replace(",,",",")
#         stuff = stuff.split(' ')
#         return " ".join(stuff)
# def parse(text):
#     MIN_TOKEN = 150
#     MAX_TOKEN = 16
#     MIN_CHARS = 300
#     tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct",trust_remote_code = True)
#     text = clean_text(text)
#     print(type(text))
#     print(len(text))
#     if tokenizer:
#         num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
#     else:
#         num_tokens = len(text.split())
#     print(num_tokens)
#     # if len(text) < MIN_CHARS or num_tokens < MIN_TOKEN:
#     #     print("none o day")
#     #     return None 
#     if num_tokens > MAX_TOKEN:
#         if tokenizer:
#             truncated = tokenizer.encode(text, add_special_tokens=False)[:MAX_TOKEN]
#             text = tokenizer.decode(truncated)
#         else:
#             text = " ".join(text.split()[:MAX_TOKEN])
#     return text

# x = "I love it, so         much,fsadf wfea.,feaefawefawef,aefa wfafew,efaw   afwef,,,,efawefawefaw"
# y = parse(x)
# print(y)