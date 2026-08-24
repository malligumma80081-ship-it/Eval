from app.evaluation.llm_judge import LLMJudge


judge = LLMJudge(
    model="llama3.2"
)


question = """
What is Python?
"""


answer = """
Python is a high-level programming language
used for software development, automation,
data analysis, and machine learning.
"""


result = judge.judge(
    question=question,
    answer=answer
)


print("\n==============================")
print("LLM AS A JUDGE")
print("==============================")


print("\nQuestion:")
print(question)


print("\nAnswer:")
print(answer)


print("\nEvaluation:")
print(result)