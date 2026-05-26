from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types
import os

app = FastAPI(title="Supervisor Assistant API")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

VECTORDB_DIR = "./data/vectordb"
chroma_client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="student_profiles", embedding_function=embedding_fn)

class SupervisorQuery(BaseModel):
    message: str

@app.post("/api/supervisor/chat")
def supervisor_chat(req: SupervisorQuery):
    # 1. Retrieve the most relevant students from the database based on the query
    results = collection.query(
        query_texts=[req.message],
        n_results=3, # Bring back the top 3 most relevant student profiles
        include=["documents", "metadatas", "distances"]
    )

    context_chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if dist < 0.85:  # Relevance threshold
                context_chunks.append(doc)

    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    prompt = """You are an elite academic administrative assistant for university supervisors.
Your job is to review the provided undergraduate data and the machine learning risk predictions, then help the supervisor identify struggling students or recommend interventions.

Guidelines:
- Base your advice STRICTLY on the provided student context.
- If a student has a high predicted risk percentage, suggest a concrete academic intervention (e.g., tutoring, a schedule review).
- Maintain a highly professional, academic tone suitable for university faculty.

Student Context Data (From Database):\n"""

    if context_text:
        prompt += context_text
    else:
        prompt += "[No relevant students found matching this query.]"

    prompt += f"\n\n---\n\nSupervisor Query: {req.message}\n"

    try:
        response = gemini_client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, # Low temperature for factual, reliable administrative advice
                max_output_tokens=1024,
            )
        )
        answer = response.text
    except Exception as e:
        answer = f"System Error: Could not reach AI services. {e}"

    return {
        "answer": answer,
        "students_referenced": [meta.get("student_id") for meta in results["metadatas"][0]] if results["metadatas"] else []
    }