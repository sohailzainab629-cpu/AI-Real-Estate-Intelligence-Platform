import os

from dotenv import load_dotenv
from groq import Groq

from app.rag import get_rag_context


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_ai(question):
    context = get_rag_context(question)

    prompt = f"""
You are an AI Real Estate Assistant.

Answer the user's question using the provided real estate
document context whenever it is relevant.

If the answer is based on the documents, mention the source
document name.

If the documents do not contain enough information, clearly
say that and then provide general real estate guidance.

Keep the answer simple, clear, practical, and beginner-friendly.

REAL ESTATE DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI real estate assistant with RAG capabilities."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content