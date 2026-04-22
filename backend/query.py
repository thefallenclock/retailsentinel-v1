import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
TOP_K = int(os.getenv('TOP_K', 3))

embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

llm = ChatGroq(
    model='llama-3.1-8b-instant',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0
)

SYSTEM_PROMPT = """You are RetailSentinel, an AI assistant for retail
policy documents. Answer ONLY using the context provided below.
If the answer is not in the context, respond EXACTLY with:
'I cannot find a reliable answer in the available documents.'
Always end your answer with: Source: [document name], Page [X]"""

def answer_query(user_query: str) -> dict:
    start = time.time()

    results = vectordb.similarity_search_with_score(user_query, k=TOP_K)

    if not results:
        return {
            'answer': 'I cannot find a reliable answer in the available documents.',
            'citations': [],
            'latency_ms': int((time.time() - start) * 1000)
        }

    context = ''
    citations = []
    for doc, score in results:
        src = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '?')
        fname = os.path.basename(src)
        context += f'[{fname}, Page {page}]\n{doc.page_content}\n\n'
        citations.append({
            'source': fname,
            'page': page,
            'score': round(float(score), 3)
        })

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f'CONTEXT:\n{context}\nQUESTION: {user_query}')
    ]

    response = llm.invoke(messages)
    latency = int((time.time() - start) * 1000)

    return {
        'answer': response.content,
        'citations': citations,
        'latency_ms': latency
    }

if __name__ == '__main__':
    test_query = "What is the return policy for electronics purchased during a festive sale?"
    result = answer_query(test_query)
    print("\n--- ANSWER ---")
    print(result['answer'])
    print("\n--- CITATIONS ---")
    for c in result['citations']:
        print(f"Source: {c['source']} | Page: {c['page']} | Score: {c['score']}")
    print(f"\nLatency: {result['latency_ms']}ms")
