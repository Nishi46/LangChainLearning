from langchain.tools import tool 
from langchain.chat_models import init_chat_model 
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator 
from langchain.messages import SystemMessage
from langchain.messages import ToolMessage
from typing import Literal 
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-5-nano",
    temperature=0
)

#define tools
@tool
def multiply(a: int, b: int) -> int:
    """multiply 'a' and 'b'."
    
    Args:
        a: first int
        b: second int 
    """
    return a * b

@tool 
def add(a: int, b: int) -> int:
    """ Adds 'a' and 'b'.

    Args:
        a: first int
        b: second int 
    """

    return a+b

@tool 
def divide(a: int, b:int) -> float:
    """ divide 'a' and 'b'.

    Args:
    a: first int
    b: second int 
    """

    return a/b

#augment the LLM with tools
tools = [add, multiply ,divide]
tools_by_name={tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


#graph state stores the messages and the number of LLM calls
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int 

#model node calls the LLM and decides whether to call a tool 
def llm_call(state:dict):
    """LLM decides whether to call a tool or not  """
    
    return {
        "messages": [
            model_with_tools.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                )
            ]
            + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls',0) + 1
        }

#tool node calls the tools and returns the results 
def tool_node(state:dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


#routes to the tool node or to the end
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    #if the LLM makes a tool call, then perform an action 
    if last_message.tool_calls:
        return "tool_node"

    #otherwise, stop 
    return END 

#build workflow 
agent_builder = StateGraph(MessagesState)

#add nodes 
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

#add edges to connect the nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

#compile the agent
agent = agent_builder.compile()

#run the agent
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()