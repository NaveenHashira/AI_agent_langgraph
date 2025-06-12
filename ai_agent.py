import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

# Tool APIs
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults


load_dotenv()

# Define the reducer
def add_messages(x: list[AnyMessage], y: list[AnyMessage]) -> list[AnyMessage]:
    return x + y

# 1. Define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 2. Tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract the second integer from the first and return the result."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide the first integer by the second and return the result as a float. Raises an error if dividing by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Return the remainder when the first integer is divided by the second."""
    return a % b

# 3. External knowledge tools
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
search = DuckDuckGoSearchResults()

tools = [multiply, add, subtract, divide, modulus, arxiv, wiki, search]

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

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 7. Run function
def run_agent(query: str, thread_id: str = "default"):
    system = HumanMessage(content="Always provide only the final answer, without any reasoning or additional text.")
    messages = [system, HumanMessage(content=query)]
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"messages": messages}, config=config)

