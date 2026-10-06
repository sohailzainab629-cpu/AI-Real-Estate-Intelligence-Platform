import os

from app.rag import (
    extract_pdf_text,
    create_chunks,
    create_vectorizer,
    add_documents
)


DOCUMENTS_FOLDER = "documents"


print("=" * 50)
print("REAL ESTATE RAG DOCUMENT INGESTION")
print("=" * 50)


# ============================================================
# CHECK DOCUMENTS FOLDER
# ============================================================

if not os.path.exists(
    DOCUMENTS_FOLDER
):

    print(
        "documents folder not found!"
    )

    raise SystemExit


# ============================================================
# FIND PDF FILES
# ============================================================

pdf_files = [

    file

    for file in os.listdir(
        DOCUMENTS_FOLDER
    )

    if file.lower().endswith(".pdf")
]


if not pdf_files:

    print(
        "No PDF files found in documents folder."
    )

    raise SystemExit


print(
    f"\nFound {len(pdf_files)} PDF file(s).\n"
)


# ============================================================
# READ ALL PDFS FIRST
# ============================================================

all_chunks = []
all_metadatas = []
all_ids = []


for filename in pdf_files:

    pdf_path = os.path.join(
        DOCUMENTS_FOLDER,
        filename
    )

    print(
        f"Processing: {filename}"
    )

    try:

        text = extract_pdf_text(
            pdf_path
        )

        if not text.strip():

            print(
                "Error: PDF contains no text."
            )

            continue

        chunks = create_chunks(
            text
        )

        print(
            f"Chunks created: {len(chunks)}"
        )

        for index, chunk in enumerate(
            chunks
        ):

            all_chunks.append(
                chunk
            )

            all_metadatas.append(
                {
                    "source": filename,
                    "chunk": index
                }
            )

            all_ids.append(
                f"{filename}_{index}"
            )

    except Exception as e:

        print(
            f"Error processing {filename}: {e}"
        )


# ============================================================
# CHECK CHUNKS
# ============================================================

if not all_chunks:

    print(
        "\nNo chunks were created."
    )

    raise SystemExit


print("\n" + "=" * 50)

print(
    f"Total chunks created: {len(all_chunks)}"
)

print("=" * 50)


# ============================================================
# CREATE ONE SHARED TF-IDF VECTORIZER
# ============================================================

print(
    "\nCreating shared TF-IDF vectorizer..."
)

vectorizer = create_vectorizer(
    all_chunks
)

print(
    "Vectorizer created successfully!"
)


# ============================================================
# STORE EVERYTHING IN CHROMADB
# ============================================================

print(
    "\nStoring documents in ChromaDB..."
)

add_documents(

    chunks=all_chunks,

    metadatas=all_metadatas,

    ids=all_ids,

    vectorizer=vectorizer
)


print("\n" + "=" * 50)

print(
    "RAG INGESTION COMPLETED"
)

print("=" * 50)

print(
    f"Total chunks stored: {len(all_chunks)}"
)

print(
    "Embedding method: Local TF-IDF"
)

print(
    "ChromaDB automatic embeddings: Disabled"
)

print("=" * 50)