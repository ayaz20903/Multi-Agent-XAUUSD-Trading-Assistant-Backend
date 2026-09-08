def retrieve_documents(vector_store, query, k=3, max_distance=0.8):
    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    filtered_results = [
        (document, score)
        for document, score in results
        if score <= max_distance
    ]

    return filtered_results