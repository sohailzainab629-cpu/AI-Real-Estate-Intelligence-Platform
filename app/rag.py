import os
import pickle

import chromadb
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


CHROMA_PATH = "./chroma_db"
VECTOR_FILE = "./chroma_db/tfidf_vectorizer.pkl"


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

# IMPORTANT:
# embedding_function=None prevents Chroma from automatically
# using all-MiniLM-L6-v2.
collection = chroma_client.get_or_create_collection(
    name="real_estate_documents",
    embedding_function=None
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text, chunk_size=200):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# CREATE TF-IDF VECTORIZER
# ============================================================

def create_vectorizer(all_chunks):

    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english"
    )

    vectorizer.fit(all_chunks)

    os.makedirs(
        os.path.dirname(VECTOR_FILE),
        exist_ok=True
    )

    with open(
        VECTOR_FILE,
        "wb"
    ) as file:

        pickle.dump(
            vectorizer,
            file
        )

    return vectorizer


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks,
    vectorizer
):

    vectors = vectorizer.transform(
        chunks
    )

    vectors = normalize(
        vectors
    )

    embeddings = []

    for vector in vectors:

        embeddings.append(
            vector.toarray()[0].tolist()
        )

    return embeddings


# ============================================================
# ADD ALL DOCUMENTS
# ============================================================

def add_documents(
    chunks,
    metadatas,
    ids,
    vectorizer
):

    embeddings = create_embeddings(
        chunks,
        vectorizer
    )

    collection.upsert(

        ids=ids,

        documents=chunks,

        embeddings=embeddings,

        metadatas=metadatas
    )


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question,
    top_k=3
):

    if not os.path.exists(
        VECTOR_FILE
    ):
        return []

    with open(
        VECTOR_FILE,
        "rb"
    ) as file:

        vectorizer = pickle.load(
            file
        )

    question_vector = vectorizer.transform(
        [question]
    )

    question_vector = normalize(
        question_vector
    )

    question_embedding = (
        question_vector
        .toarray()[0]
        .tolist()
    )

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    final_results = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        final_results.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown"
                )
            }
        )

    return final_results


# ============================================================
# GET RAG CONTEXT
# ============================================================

def get_rag_context(
    question,
    top_k=3
):

    results = search_documents(
        question,
        top_k
    )

    if not results:

        return "No relevant documents found."

    context = []

    for result in results:

        context.append(
            f"Source: {result['source']}\n"
            f"{result['text']}"
        )

    return "\n\n".join(
        context
    )