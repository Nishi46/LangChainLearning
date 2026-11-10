from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("Understanding the message types in LangChain")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )

    #structured messages for better control 
    messages = [
        SystemMessage(content="You are a helpful AI assistant who explains things simply."),
        HumanMessage(content="Explain quantum computing to a 10-year-old."),
    ]

    response = model.invoke(messages)
   
    print("🤖 AI Response:\n")
    print(response.content)
    print("\n✅ Notice how the SystemMessage influenced the response style!")

if __name__ == "__main__":
    main()