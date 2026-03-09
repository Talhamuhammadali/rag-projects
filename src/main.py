"""Main module for rag-projects."""
import asyncio
from dotenv import load_dotenv, find_dotenv

from utils.llm import ainvoke

load_dotenv(find_dotenv()) 

async def main():
    response = await ainvoke(
        [
            {"role": "user", "content": "What are some fun things to do in New York? keep it short."}
        ]
    )
    print("LLM Response:", response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
