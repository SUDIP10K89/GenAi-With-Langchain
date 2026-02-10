from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9,max_completion_tokens=100)
result = model.invoke("What is the capital of France?")

print(result.content)