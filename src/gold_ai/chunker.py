import re

from langchain_core.documents import Document


def chunk_documents(documents):
    chunks = []

    for document in documents:
        sections = re.split(
            r"(?=^\d+\.\s)",
            document.page_content,
            flags=re.MULTILINE
        )

        for section in sections:
            section = section.strip()

            if not section:
                continue

            chunk = Document(
                page_content=section,
                metadata=document.metadata
            )

            chunks.append(chunk)

            print("\n--- CHUNK ---")
            print(chunk.page_content)
            print("-------------")

    return chunks