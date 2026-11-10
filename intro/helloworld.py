from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
import os

load_dotenv()

def main():
  
  #create the AI model 
  model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY")
  )

  #first api call 
  response = model.invoke("What is LangChain in one sentence?")
  
  print("🤖 AI Response:", response.content)
  print("\n✅ Success! You just made your first LangChain call!")

if __name__ == "__main__":
    main()