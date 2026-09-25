import os
from typing import Union, List, Dict, Any
from google import genai
from dotenv import load_dotenv

load_dotenv()


def generate_answer(user_message: str, context: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Generates a grounded response to the user's message using retrieved context from ChromaDB.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."

    client = genai.Client(api_key=api_key)

    # Format context if passed as retrieved chunks
    if isinstance(context, list):
        if not context:
            return "I couldn't find any relevant admission or syllabus information in the knowledge base for your query."

        formatted_chunks = []
        for i, chunk in enumerate(context):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Document")
            page = meta.get("page", "?")
            text = chunk.get("text", "")
            formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
        context_str = "\n\n".join(formatted_chunks)
    else:
        context_str = str(context)

    prompt = f"""You are the Admission AI Assistant for PGDCA and Computer Science / M.Sc.IT programs at Gujarat Vidyapith.
Answer the user's question clearly, politely, and accurately based ONLY on the provided context.

Guidelines:
1. Base your answer strictly on the Context below. Do not make up facts.
2. If the context does not contain enough information to answer the question, clearly state that the provided admission documents do not cover that information and offer guidance on what is available.
3. Keep the tone professional, welcoming, and academic. Use concise formatting or bullet points when explaining course outlines, credits, or requirements.

Context:
{context_str}

User Question:
{user_message}
"""

    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.7-flash")

    try:
        response = client.interactions.create(
            model=model_name,
            input=prompt,
        )
        return response.output_text
    except Exception as e:
        # Fallback to standard generate_content if interactions fails
        try:
            fallback_model = "gemini-2.5-flash"
            fb_response = client.models.generate_content(
                model=fallback_model,
                contents=prompt,
            )
            return fb_response.text
        except Exception as inner_e:
            return f"An error occurred while generating the answer: {str(e)}"