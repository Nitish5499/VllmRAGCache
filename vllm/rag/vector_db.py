from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

@dataclass
class RetrievedDocument:
    content: str
    metadata: Dict[str, Any]
    score: float

class VectorDBConnector(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Retrieve relevant documents for the query"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the vector store"""
        pass

class ChromaDBConnector(VectorDBConnector):
    def __init__(
        self,
        collection_name: str = "vllm_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        # Initialize ChromaDB client
        settings = Settings()
        if persist_directory:
            settings = Settings(persist_directory=persist_directory,
                             is_persistent=True)
        
        self.client = chromadb.Client(settings)
        
        # Set up embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
        except ValueError:
            # Collection already exists
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to ChromaDB"""
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
            
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Retrieve relevant documents from ChromaDB"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieved_docs = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Convert distance to similarity score (ChromaDB returns L2 distance)
            similarity_score = 1.0 / (1.0 + distance)
            retrieved_docs.append(
                RetrievedDocument(
                    content=doc,
                    metadata=metadata,
                    score=similarity_score
                )
            )
        
        return retrieved_docs

class RAGProcessor:
    def __init__(self, vector_db: VectorDBConnector):
        self.vector_db = vector_db
    
    def augment_prompt(self, original_prompt: str, top_k: int = 3) -> str:
        """Augment the prompt with retrieved context"""
        # Retrieve relevant documents
        retrieved_docs = self.vector_db.retrieve(original_prompt, top_k)
        
        # Format context with relevance scores
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_part = (
                f"Context {i+1} (Relevance: {doc.score:.2f}):\n"
                f"{doc.content}"
            )
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Combine context with original prompt
        augmented_prompt = (
            f"Given the following relevant context:\n\n{context}\n\n"
            f"Question: {original_prompt}\n\n"
            "Please provide a detailed answer based on the context above. "
            "If the context doesn't contain relevant information, please indicate that."
        )
        
        return augmented_prompt
