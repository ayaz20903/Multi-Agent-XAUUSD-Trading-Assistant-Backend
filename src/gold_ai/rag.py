
from langchain_groq import ChatGroq
from gold_ai.config import GROQ_API_KEY

_llm_instance = None


def create_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=GROQ_API_KEY,
            temperature=0,
        )
    return _llm_instance

def generate_answer(llm, query, documents):
    context = "\n\n".join(
        document.page_content
        for document, score in documents
    )

    prompt = f"""
        You are an AI assistant answering questions about my personal XAUUSD trading strategy.

        IMPORTANT RULES:
        1. Answer ONLY using the provided context.
        2. Do NOT use your general knowledge or assumptions.
        3. Do NOT invent or guess an answer.
        4. If the answer is explicitly stated in the context, use that exact information.
        5. If the answer cannot be found in the context, say:
        "I don't have enough information in my trading knowledge base."

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        return response.content

    return "\n".join(
        item["text"]
        for item in response.content
        if item.get("type") == "text"
    )
