"""Fixed questions in a pipeline that routes on the answer. Run: uv run python examples/pipeline.py"""

from langchain_core.runnables import RunnableLambda

from langchain_wity import SystemOneResponse, WityClassifier, choice

team = WityClassifier(
    questions={
        "team": choice(
            "Which team should handle this ticket?",
            {"billing": "Payments, charges and refunds", "technical": "Bugs, errors and outages"},
        )
    },
    reasoning="off",
)


def send(res: SystemOneResponse) -> str:
    answer = res.choices["team"]
    if answer.confidence < 0.7:
        return "Sent to a person: Wity isn't sure."
    return f"Sent to {answer.choice}."


pipeline = RunnableLambda(lambda ticket: ticket.strip()) | team | send

print(pipeline.invoke("  The app crashes when I upload a photo.  "))
print(pipeline.batch(["Please refund my last invoice.", "Login page shows error 500."]))
