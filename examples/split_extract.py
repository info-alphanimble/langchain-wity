"""Cut an email thread into emails, then pull fields out of each. Run: uv run python examples/split_extract.py"""

from langchain_core.documents import Document

from langchain_wity import WityExtractor, WitySplitter

THREAD = """From: dana@example.com
The seal test on batch 2231 failed at 40 psi. Can we rerun it before Friday?

From: marcus@example.com
Yes. I booked the rig for Thursday 9 October at 10:00. It takes about 3 hours.
"""

splitter = WitySplitter(split="one item per email")
extractor = WityExtractor(
    fields={"sender": "The email address the email is from", "when": "A date the email mentions"},
    types={"when": "date"},
)

for email in splitter.split_documents([Document(page_content=THREAD, metadata={"source": "thread-2231"})]):
    res = extractor.invoke(email)
    print(email.metadata, res.fields)
