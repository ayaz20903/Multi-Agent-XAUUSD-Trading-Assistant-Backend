from langchain_chroma import Chroma


def create_vector_store(chunks, embeddings, vectors):
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    vector_store = Chroma(
        collection_name="gold_strategy",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=vectors
    )

    return vector_store


def load_vector_store(embeddings):
    return Chroma(
        collection_name="gold_strategy",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )