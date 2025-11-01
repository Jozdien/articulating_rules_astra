from collections import Counter


class Evaluator:
    def __init__(self, mode: str, client=None, eval_model: str = None, classifier_template: str = None):
        self.mode = mode
        self.client = client
        self.eval_model = eval_model
        self.classifier_template = classifier_template

    def evaluate_frequency(self, responses: list[str]) -> dict:
        counts = Counter(responses)
        total = len(responses)
        return {response: count / total for response, count in counts.items()}

    def evaluate_with_classifier(self, question: str, response: str, temperature: float = 0.0) -> str:
        prompt = self.classifier_template
        if "{{QUESTION}}" in prompt:
            prompt = prompt.replace("{{QUESTION}}", question)
        if "{{RESPONSE}}" in prompt:
            prompt = prompt.replace("{{RESPONSE}}", response)

        messages = [{"role": "user", "content": prompt}]
        return self.client.query(self.eval_model, messages, temperature)

    def aggregate_classifier_results(self, results: list[str]) -> dict:
        counts = Counter(r.strip().lower() for r in results)
        total = len(results)
        return {result: count / total for result, count in counts.items()}
