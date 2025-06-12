import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.agents import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_groq import ChatGroq

# Tool APIs
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.arxiv.utils import ArxivAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.wikipedia.utils import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. Define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], lambda x: x]

# 2. Tools
@tool
def multiply(a: int, b: int) -> int:
    return a * b

@tool
def add(a: int, b: int) -> int:
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    return a - b

@tool
def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    return a % b

# 3. External knowledge tools
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
tavily = TavilySearchResults()

tools = [multiply, add, subtract, divide, modulus, arxiv, wiki, tavily]

# 4. LLM + Tool binding
llm = ChatGroq(model="qwen-qwq-32b", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools(tools=tools)

# 5. Node function
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 6. Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

graph = builder.compile()

# 7. Run function
def run_agent(query: str):
    return graph.invoke({"messages": [HumanMessage(content=query)]})
