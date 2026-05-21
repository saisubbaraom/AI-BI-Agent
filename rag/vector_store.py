from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

class SimpleDocumentStore:
    """
    A lightweight, pure-Python document store using BM25 keyword-based
    retrieval for indexing and querying analysis reports and metadata.
    """
    def __init__(self):
        self.documents = []
        self.retriever = None

    def add_text(self, text: str, metadata: dict = None):
        """Adds a text block to the document store."""
        if text and len(text.strip()) > 0:
            doc = Document(page_content=text, metadata=metadata or {})
            self.documents.append(doc)
            # Reset retriever since corpus changed
            self.retriever = None

    def add_document(self, doc: Document):
        """Adds an existing Document object."""
        self.documents.append(doc)
        self.retriever = None

    def build_index(self):
        """Initializes the BM25 search index."""
        if not self.documents:
            return
        
        # Build retriever from current document collection
        self.retriever = BM25Retriever.from_documents(self.documents)
        # Configure to return top 3 matching chunks
        self.retriever.k = 3

    def query(self, query_str: str) -> list[Document]:
        """Queries the document store and returns relevant documents."""
        if not self.documents:
            return []
        
        if self.retriever is None:
            self.build_index()
            
        if self.retriever:
            try:
                return self.retriever.invoke(query_str)
            except Exception as e:
                print(f"Retrieval error: {e}")
                # Fallback to returning first few documents
                return self.documents[:3]
        return []
