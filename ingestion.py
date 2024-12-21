from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import time

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_docs)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    batch_size = 50
    total_batches = len(documents) // batch_size + 1
    failed_batches = []

    for i in range(total_batches):
        batch = documents[i * batch_size : (i + 1) * batch_size]
        try:
            PineconeVectorStore.from_documents(
                batch, embeddings, index_name="langchain-doc-index-v2"
            )
            print(f"Successfully added batch {i+1}/{total_batches}")
        except Exception as e:
            print(f"Error adding batch {i+1}/{total_batches}: {e}")
            failed_batches.append(i)

    # Retry failed batches
    print("Retrying failed batches...")
    for batch_index in failed_batches:
        batch = documents[batch_index * batch_size : (batch_index + 1) * batch_size]
        try:
            PineconeVectorStore.from_documents(
                batch, embeddings, index_name="langchain-doc-index-v2"
            )
            print(f"Successfully retried batch {batch_index+1}/{total_batches}")
        except Exception as e:
            print(f"Failed again for batch {batch_index+1}/{total_batches}: {e}")

    print("Done loading vectors to the vector store...")


if __name__ == "__main__":
    ingest_docs()
