from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEvaluator:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(
            model_name
        )


    def similarity(
        self,
        expected,
        actual
    ):

        embeddings = self.model.encode(
            [
                expected,
                actual
            ]
        )

        score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(score)


    def evaluate(
        self,
        expected,
        actual,
        threshold=0.80
    ):

        score = self.similarity(
            expected,
            actual
        )

        passed = score >= threshold

        return {
            "score": score,
            "threshold": threshold,
            "passed": passed
        }