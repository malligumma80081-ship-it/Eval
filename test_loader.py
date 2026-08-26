from app.rag.document_loader import load_documents


documents = load_documents(
    "data/documents"
)


for document in documents:

    print("\nSOURCE:")
    print(document["source"])

    print("\nTEXT:")
    print(document["text"])