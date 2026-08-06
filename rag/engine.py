import os
import sys
import site
import time
from typing import Dict, Any, List, Tuple

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.config import Config
from rag.document_loader import CVDocumentLoader
from rag.vector_store import VectorStoreManager

class CVChatEngine:
    """
    RAG Conversational Engine for querying Candidate CV documents.
    Combines vector retrieval with ChatNVIDIA LLM to provide grounded, accurate answers.
    """

    def __init__(self, auto_index: bool = True):
        """
        Initializes vector database, document loader, and NVIDIA Chat LLM.

        Args:
            auto_index (bool): If True, automatically scans document directory and builds vector DB if empty.
        """
        Config.validate()

        # Initialize Vector Database Manager
        self.vector_manager = VectorStoreManager()
        self.loader = CVDocumentLoader()

        # Auto-index documents if vector store is empty or forced
        if auto_index:
            self.ensure_indexed()

        # Initialize NVIDIA Chat LLM Endpoint (Llama 3.3 70B Instruct)
        self.llm = ChatNVIDIA(
            model=Config.LLM_MODEL,
            api_key=Config.NVIDIA_API_KEY,
            temperature=0.2,  # Low temperature for factual consistency with resume content
            top_p=0.7,
            max_tokens=1024,
            timeout=Config.REQUEST_TIMEOUT
        )

        # Build custom RAG Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an intelligent, professional AI Assistant designed to answer questions about a candidate's CV/Resume. "
             "Your role is to help recruiters, hiring managers, and interviewers understand the candidate's qualifications, skills, experience, education, and achievements.\n\n"
             "Guidelines:\n"
             "1. Base your answer strictly on the retrieved CV context provided below.\n"
             "2. Provide comprehensive, direct, and well-structured answers incorporating all details present in the context (roles, companies, dates, skills, projects, degrees).\n"
             "3. Pay strict attention to job titles, company names, and dates. Ensure bullet points are attributed to the exact company or role that precedes them. Do NOT attribute a top snippet bullet to a new job title that appears after it.\n"
             "4. If a specific detail is not present in the provided CV context, state that clearly without guessing.\n"
             "5. Do not make up or extrapolate facts that are not supported by the CV text.\n\n"
             "Retrieved CV Context:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    def ensure_indexed(self, force_reindex: bool = False):
        """
        Ensures documents are indexed into the vector database.
        If the vector database is empty or force_reindex is True, loads documents and indexes them.
        """
        count = self.vector_manager.get_document_count()
        if count == 0 or force_reindex:
            print("[RAG Engine] Indexing CV documents into vector store...")
            chunks = self.loader.load_and_split()
            if chunks:
                self.vector_manager.build_vector_store(chunks, clear_existing=True)
            else:
                print("[RAG Engine] Warning: No documents found to index.")
        else:
            print(f"[RAG Engine] Vector DB already contains {count} indexed document vectors.")

    def format_docs(self, docs: List[Any]) -> str:
        """
        Helper function to concatenate retrieved document contents into a unified context string.
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "CV Document")
            formatted.append(f"--- Document Snippet {i} (Source: {os.path.basename(source)}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    def query(self, question: str, chat_history: List[Tuple[str, str]] = None, k: int = None) -> Dict[str, Any]:
        """
        Processes a user question through the RAG pipeline.

        Step-by-Step RAG Execution:
        1. Query Vector Search: Searches ChromaDB using NVIDIA embeddings for top-k relevant CV snippets.
        2. Context Assembly: Combines snippets into formatted prompt context.
        3. LLM Prompt Construction: Injects context, chat history, and user question into prompt template.
        4. Generation: Calls ChatNVIDIA to generate grounded, fact-checked response.

        Args:
            question (str): User query (e.g., "What programming languages does the candidate know?").
            chat_history (List[Tuple[str, str]], optional): Previous chat messages [(human, ai)].
            k (int): Number of top context chunks to retrieve.

        Returns:
            Dict[str, Any]: Dictionary containing 'answer', 'sources', and 'retrieved_docs'.
        """
        target_k = k or Config.RETRIEVAL_K
        chat_history_messages = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                chat_history_messages.append(("human", human_msg))
                chat_history_messages.append(("ai", ai_msg))

        # Retrieve relevant CV snippets using similarity search
        raw_retrieved = self.vector_manager.similarity_search(query=question, k=target_k)

        # Deduplicate retrieved snippets while preserving order
        seen_texts = set()
        retrieved_docs = []
        for doc in raw_retrieved:
            txt = doc.page_content.strip()
            if txt not in seen_texts:
                seen_texts.add(txt)
                retrieved_docs.append(doc)

        # Format context string
        context_str = self.format_docs(retrieved_docs)

        # Generate response using LLM chain with network retry protection
        chain = self.prompt | self.llm | StrOutputParser()
        
        max_attempts = Config.MAX_RETRIES
        last_exception = None
        answer = None

        for attempt in range(1, max_attempts + 1):
            try:
                answer = chain.invoke({
                    "context": context_str,
                    "chat_history": chat_history_messages,
                    "question": question
                })
                break
            except Exception as e:
                last_exception = e
                delay = 2 ** attempt
                print(f"[RAG Engine] Network request attempt {attempt}/{max_attempts} failed ({e}). Retrying in {delay}s...")
                time.sleep(delay)

        if answer is None:
            answer = (
                "**Service Notice**: The AI service is currently busy or rate-limited. Please wait a moment and try your question again."
            )

        # Extract source metadata
        sources = list(set([os.path.basename(doc.metadata.get("source", "CV")) for doc in retrieved_docs]))

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs
        }

    def query_stream(self, question: str, chat_history: List[Tuple[str, str]] = None, k: int = None):
        """
        Streams RAG answer tokens in real-time using ChatNVIDIA streaming generator.

        Returns:
            Tuple[generator, list, list]: (stream_generator, sources, retrieved_docs)
        """
        target_k = k or Config.RETRIEVAL_K
        chat_history_messages = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                chat_history_messages.append(("human", human_msg))
                chat_history_messages.append(("ai", ai_msg))

        raw_retrieved = self.vector_manager.similarity_search(query=question, k=target_k)
        seen_texts = set()
        retrieved_docs = []
        for doc in raw_retrieved:
            txt = doc.page_content.strip()
            if txt not in seen_texts:
                seen_texts.add(txt)
                retrieved_docs.append(doc)

        context_str = self.format_docs(retrieved_docs)
        chain = self.prompt | self.llm | StrOutputParser()
        sources = list(set([os.path.basename(doc.metadata.get("source", "CV")) for doc in retrieved_docs]))

        def stream_generator():
            max_attempts = Config.MAX_RETRIES
            for attempt in range(1, max_attempts + 1):
                try:
                    stream = chain.stream({
                        "context": context_str,
                        "chat_history": chat_history_messages,
                        "question": question
                    })
                    for chunk in stream:
                        if chunk:
                            yield str(chunk)
                    return
                except Exception as e:
                    print(f"[RAG Engine Stream] Attempt {attempt}/{max_attempts} failed ({e})...")
                    if attempt < max_attempts:
                        delay = 2 ** attempt
                        time.sleep(delay)
                    else:
                        yield "\n\n**Service Notice**: The AI service is currently busy or rate-limited. Please wait a moment and try your question again."

        return stream_generator(), sources, retrieved_docs
