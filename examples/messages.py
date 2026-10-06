"""Use a chat as the state. Run: uv run python examples/messages.py"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from langchain_wity import WityClassifier, noul

upset = WityClassifier(questions={"upset": noul("Is the customer upset?")})

chat = [
    HumanMessage("Hi, my order hasn't arrived."),
    AIMessage("Sorry about that! It should arrive tomorrow."),
    HumanMessage("You said that last week. This is the third time."),
]

print(upset.invoke(chat).nouls["upset"].noul)


async def main() -> None:
    # Messages can also sit inside a dict, next to other data.
    res = await upset.ainvoke({"customer": {"plan": "pro", "orders": 14}, "chat": chat})
    print(res.nouls["upset"].noul)


asyncio.run(main())
