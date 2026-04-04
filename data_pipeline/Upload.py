import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()


def load_documents(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix in {".md", ".txt"}:
        return TextLoader(str(file_path), encoding="utf-8").load()

    try:
        from langchain_unstructured import UnstructuredLoader  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Unsupported file type for TextLoader. Install langchain-unstructured:\n"
            "  python -m pip install langchain-unstructured\n"
        ) from e

    return UnstructuredLoader(str(file_path)).load()


def main():
    parser = argparse.ArgumentParser(description="Load a file and upload chunks to Qdrant.")
    parser.add_argument(
        "--file",
        default=None,
        help="Path to the input file (.md/.txt recommended)",
    )
    parser.add_argument("--collection", default="volkswagen_newsroom")
    args = parser.parse_args()

    if not args.file:
        raise SystemExit(
            "Pass --file <path> to your document. Example:\n"
            "  python -m data_pipeline.Upload --file ./data/example.md"
        )

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    docs = load_documents(file_path)
    print(f"Loaded {len(docs)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunk(s).")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError(
            "QDRANT_URL is required (environment or .env at repo root)."
        )
    if api_key is None:
        raise RuntimeError(
            "QDRANT_API_KEY is required (environment or .env at repo root)."
        )

    QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=url,
        api_key=api_key,
        collection_name=args.collection,
    )
    print(f"Uploaded {len(splits)} chunk(s) to Qdrant collection '{args.collection}'.")


if __name__ == "__main__":
    main()
