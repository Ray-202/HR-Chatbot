import pdfplumber
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # for doc data structure
from langchain_community.vectorstores import Pinecone as PineconeStore



#1. initialize pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


Index_Name = "chatbot"
# Check if the index exists; if not, create it
if Index_Name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=Index_Name,
        dimension=1024,  
        metric='cosine',  
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  
        )
    )
#2. create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large") 
#3. define function for chucking 

#4. define function for extracting text 
def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF using pdfplumber."""
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            all_text += page_text + "\n"
    return all_text
def chunk_text_content(text_content: str, 
                       chunk_size: int = 500, 
                       chunk_overlap: int = 50) -> list[str]:
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text_content)
#5. defie function for storing into vector store in pinecone 
def store_in_pinecone(doc_id: int, chunks: list[str]):
    """
    Takes a list of chunk strings and stores them in Pinecone.
    Each chunk is turned into a LangChain Document with metadata 
    indicating the source doc_id.
    """
    docs = []
    for i, chunk_text in enumerate(chunks):
        metadata = {"source_doc_id": doc_id, "chunk_index": i}
        docs.append(Document(page_content=chunk_text, metadata=metadata))

    # Log the chunks being ingested
    print(f"Preparing to store {len(docs)} documents in Pinecone.")

    # Use PineconeStore to embed and insert the documents
    vectorstore = PineconeStore.from_documents(
        docs,
        embedding=embeddings,
        index_name=Index_Name
    )

    print(f"Stored documents in Pinecone with vectorstore: {vectorstore}")
