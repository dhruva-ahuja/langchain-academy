import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from IPython.display import display, Image

# load_dotenv()

def _get_env(var: str):
    if not os.environ.get(var):
        print(f'{var} is not defined.')
    print(f'{var} is loaded in env')

_get_env('OPENAI_API_KEY')

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOllama(model="llama3.1")
llm_with_tools = llm.bind_tools([multiply])


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)

builder.add_edge("tools", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

messages = [HumanMessage(content="Hello, what is 2 multiplied by 3?")]
messages = graph.invoke({"messages": messages})

for m in messages['messages']:
    m.pretty_print()