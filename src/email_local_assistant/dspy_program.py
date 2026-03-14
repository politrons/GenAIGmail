from __future__ import annotations

import dspy


class EmailDomainAnswer(dspy.Signature):
    """Answer user questions from email data with concise evidence-grounded output."""

    context = dspy.InputField(desc="Email assistant policy and constraints")
    question = dspy.InputField(desc="User question about emails")
    answer = dspy.OutputField(desc="Accurate and concise answer grounded in available email evidence")


class EmailAssistant(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.respond = dspy.Predict(EmailDomainAnswer)

    def forward(self, question: str, context: str):
        return self.respond(question=question, context=context)
