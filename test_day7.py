from app.rag.rag_pipeline import (
    RAGPipeline
)


rag = RAGPipeline(
    document_folder="data/documents",
    model="llama3.2"
)


question = "Who created Python?"


result = rag.ask(
    question
)


print("\nQUESTION")
print("---------")

print(
    result["question"]
)


print("\nRETRIEVED CONTEXT")
print("-----------------")

for item in result["context"]:

    print(
        f"[{item['source']}]"
    )

    print(
        item["text"]
    )


print("\nANSWER")
print("------")

print(
    result["answer"]
)


# from app.evaluation.rag_evaluator import RAGEvaluator
# from app.rag import rag_pipeline 


# # -----------------------------
# # 1. Create the RAG instance
# # -----------------------------
# rag = rag_pipeline.RAGPipeline(
#     document_folder="data/documents",
#     model="llama3.2"
# )


# # -----------------------------
# # 2. Define the question
# # -----------------------------
# question = "Who created Python?"


# # -----------------------------
# # 3. Ask the RAG system
# # -----------------------------
# result = rag.ask(question)


# # -----------------------------
# # 4. Get context
# # -----------------------------
# # Adjust this depending on what your rag.ask()
# # actually returns.
# context = result.get("context", "")


# # -----------------------------
# # 5. Create evaluator
# # -----------------------------
# evaluator = RAGEvaluator(
#     model="llama3.2"
# )


# # -----------------------------
# # 6. Evaluate context relevance
# # -----------------------------
# context_result = evaluator.evaluate_context_relevance(
#     question=question,
#     context=context
# )


# # -----------------------------
# # 7. Evaluate faithfulness
# # -----------------------------
# faithfulness_result = evaluator.evaluate_faithfulness(
#     question=question,
#     context=context,
#     answer=result["answer"]
# )


# # -----------------------------
# # 8. Evaluate answer relevance
# # -----------------------------
# relevance_result = evaluator.evaluate_answer_relevance(
#     question=question,
#     answer=result["answer"]
# )


# # -----------------------------
# # 9. Print results
# # -----------------------------
# print("\nRAG ANSWER")
# print("----------")
# print(result["answer"])

# print("\nEVALUATION")
# print("----------")

# print(
#     "Context Relevance:",
#     context_result
# )

# print(
#     "Faithfulness:",
#     faithfulness_result
# )

# print(
#     "Answer Relevance:",
#     relevance_result
# )
