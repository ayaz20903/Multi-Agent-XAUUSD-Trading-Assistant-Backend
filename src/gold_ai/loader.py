from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents():
    loader = DirectoryLoader(
        "data",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False
    )

    documents = loader.load()

    return documents