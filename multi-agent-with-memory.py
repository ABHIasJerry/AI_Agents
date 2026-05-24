import os
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --- 1. SHARED CONFIG & DATA STRUCTURES ---

class FinalAgentResponse(BaseModel):
    human_response: str = Field(description="The final warm, human-like answer back to the user.")
    source_used: str = Field(description="Which agent/tool answered this or if it was answered from memory. E.g., 'Weather Worker', 'Web Search Worker', or 'Memory Context'")

class WeatherStructure(BaseModel):
    temperature: float
    condition: str
    human_response: str

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --- 2. DEFINE WORKER TOOLS (SAME AS BEFORE) ---

def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"it's sunny and 70 degrees in {city}"

def weather_agent_worker(query: str) -> str:
    """Useful when you need to answer questions specifically about the weather, climate, or forecasts."""
    worker_llm_tools = llm.bind_tools([weather_tool])
    worker_structured_llm = llm.with_structured_output(WeatherStructure)
    
    worker_messages = [
        SystemMessage(content="You are a meteorological expert. Use your weather tool to find facts."),
        HumanMessage(content=query)
    ]
    
    ai_msg = worker_llm_tools.invoke(worker_messages)
    if ai_msg.tool_calls:
        worker_messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            if tool_call["name"] == "weather_tool":
                output = weather_tool(**tool_call["args"])
                worker_messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
        final_res = worker_structured_llm.invoke(worker_messages)
    else:
        final_res = worker_structured_llm.invoke(worker_messages)
        
    return f"Weather Worker Result -> Temp: {final_res.temperature}, Condition: {final_res.condition}. Message: {final_res.human_response}"

def search_agent_worker(query: str) -> str:
    """Useful when answering general knowledge, news, current events, or non-weather queries."""
    ddg_search = DuckDuckGoSearchRun()
    search_result = ddg_search.run(query)
    
    search_messages = [
        SystemMessage(content="You are a research expert. Synthesize raw search results into a clean summary."),
        HumanMessage(content=f"Summarize this finding to answer the user's intent: {search_result}")
    ]
    res = llm.invoke(search_messages)
    return f"Search Worker Result -> {res.content}"


# --- 3. MEMORY MANAGEMENT SETUP ---

# This dictionary stores chat histories per session ID
# Format: { "session_1": [SystemMessage, HumanMessage, AIMessage, ...] }
MEMORY_STORE = {}

def get_session_history(session_id: str) -> list:
    if session_id not in MEMORY_STORE:
        # Initialize session with a foundational System Prompt for the Supervisor
        MEMORY_STORE[session_id] = [
            SystemMessage(
                content="You are a team Supervisor. You maintain the conversation flow. "
                        "Review the conversation history to understand context. "
                        "If you need new information, delegate tasks to sub-agents (weather_agent_worker or search_agent_worker) using tools. "
                        "If the user's question references past turns, use the history context to fulfill it. "
                        "Always compile your final thoughts into the required structured schema format."
            )
        ]
    return MEMORY_STORE[session_id]


# --- 4. SUPERVISOR AGENT EXECUTION FLOW ---

agent_tools = [weather_agent_worker, search_agent_worker]
supervisor_llm_with_tools = llm.bind_tools(agent_tools)
supervisor_structured_output = llm.with_structured_output(FinalAgentResponse)

def chat_with_multi_agent(session_id: str, user_input: str) -> FinalAgentResponse:
    # 1. Fetch historical message array for this specific user session
    history = get_session_history(session_id)
    
    # 2. Append the new user turn to history
    history.append(HumanMessage(content=user_input))
    
    # 3. Present the entire conversational history to the Supervisor
    supervisor_decision = supervisor_llm_with_tools.invoke(history)
    
    # Check if Supervisor wants to call a sub-agent tool to fulfill the request
    if supervisor_decision.tool_calls:
        # Clone history to avoid cluttering main memory with internal worker thought loops
        execution_context = list(history)
        execution_context.append(supervisor_decision)
        
        for tool_call in supervisor_decision.tool_calls:
            chosen_worker = next(t for t in agent_tools if t.__name__ == tool_call["name"])
            print(f"\n[Supervisor] 🤖 Needs data. Routing task to: {chosen_worker.__name__}...")
            
            # Execute worker agent
            worker_output = chosen_worker(**tool_call["args"])
            
            execution_context.append(ToolMessage(content=worker_output, tool_call_id=tool_call["id"]))
            
        # Compile final structured response from the temporary execution history
        final_verdict = supervisor_structured_output.invoke(execution_context)
    else:
        # Supervisor can answer directly from memory context without invoking tools
        print("\n[Supervisor] 🧠 Answering directly from conversation memory...")
        final_verdict = supervisor_structured_output.invoke(history)
        
    # 4. Save the final human response back into memory history so next turns remember it
    history.append(AIMessage(content=final_verdict.human_response))
    
    return final_verdict


# --- 5. VERIFICATION (MULTI-TURN CHAT) ---
if __name__ == "__main__":
    SESSION_ID = "user_123"
    
    # Turn 1: Requires tool execution (Weather)
    print("--- TURN 1 ---")
    res1 = chat_with_multi_agent(SESSION_ID, "What is the weather like in San Francisco?")
    print(f"Agent: {res1.human_response}")
    print(f"Metadata: {repr(res1)}")

    print("\n" + "="*60 + "\n")

    # Turn 2: Tests Memory context directly (Context-dependent)
    print("--- TURN 2 (Testing Memory Integration) ---")
    res2 = chat_with_multi_agent(SESSION_ID, "What city did I just ask you about, and how is it there?")
    print(f"Agent: {res2.human_response}")
    print(f"Metadata: {repr(res2)}")
