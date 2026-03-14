"""Main module for rag-projects."""
import asyncio
from dotenv import load_dotenv, find_dotenv

from utils import ainvoke, create_vector_store, load_dataframe


load_dotenv(find_dotenv()) 

async def main():
    """Entry point: sends a test prompt to the LLM and prints the response."""
    response = await ainvoke(
        [
            {"role": "user", "content": "What are some fun things to do in New York? keep it short."}
        ],
        model="openai/gpt-oss-20b"
    )
    print("LLM Response:", response.choices[0].message.content)
    df = load_dataframe("data/news_data_dedup.csv")
    await create_vector_store(df, save_path="data/news_data_vector_store.joblib")    

    
    
    


if __name__ == "__main__":
    asyncio.run(main())
