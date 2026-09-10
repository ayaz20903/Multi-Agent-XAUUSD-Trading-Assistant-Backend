from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_documents():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )

    documents = loader.load()

    return documents