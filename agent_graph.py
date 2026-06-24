from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from agent import *


class AgentState(TypedDict):
    question: str
    is_English_question: EnglishCheckResult


linguist = Linguist(
    config={
        "BASE_LLM_MODEL": "openai:gpt-4.1-mini"
    }
)

check_english_node = linguist.as_langgraph_node(
    action="initial",
    output_key="is_English_question",
    input_mapper=lambda state: {
        "prompt": state["question"]
    },
    output_schema=EnglishCheckResult,
    max_retry_times=3,
)

graph = StateGraph(AgentState)

graph.add_node("check_english", check_english_node)
graph.add_edge(START, "check_english")
graph.add_edge("check_english", END)

app = graph.compile()

result = app.invoke({
    "question": "What is machine learning?"
})

print(result)