"""Ask Wity three questions about a support ticket. Run: uv run python examples/quick_start.py"""

from langchain_wity import WityClassifier, choice, noul, score

classifier = WityClassifier()

res = classifier.invoke(
    {
        "state": "I was charged twice for my subscription this month. Please refund one of them today.",
        "questions": {
            "team": choice(
                "Which team should handle this ticket?",
                {
                    "billing": "Payments, charges and refunds",
                    "technical": "Bugs, errors and outages",
                    "other": "Anything else",
                },
            ),
            "urgent": noul(
                "Does the customer ask for something to happen soon?",
                yes="They ask for fast action",
                no="No time pressure",
            ),
            "severity": score("How much is this hurting the customer?", ["Not at all", "A little", "A lot"]),
        },
    }
)

print(res.choices["team"].choice)
print(res.choices["team"].probabilities)
print(res.nouls["urgent"].noul)
print(res.scores["severity"].score)
