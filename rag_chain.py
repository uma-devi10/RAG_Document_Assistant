from langchain_ollama import OllamaLLM

def ask_question(vector_db, question):

    docs = vector_db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    llm = OllamaLLM(model="llama3")

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information from the document context.
If the answer is not present, say "Answer not found in document".



Context:
{context}

Question:
{question}

Answer:
"""

    result = llm.invoke(prompt)
    # `llm(...)` may return a plain string or a LangChain LLMResult-like object.
    if isinstance(result, str):
        return result
    if hasattr(result, "generations") and result.generations:
        try:
            return result.generations[0][0].text
        except Exception:
            return str(result)

    return str(result)