from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
    max_new_tokens=128,
)

chat = ChatHuggingFace(llm=llm)

result = chat.invoke("What is the capital of Nepal?")

print(result)