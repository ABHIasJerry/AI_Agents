import os
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --- 1. SHARED CONFIG & DATA STRUCTURES ---

# Final expected output structure for the user
class FinalAgentResponse(BaseModel):
    human_response: str = Field(description="The final warm, human-like answer back to the user.")
    source_used: str = Field(description="Which agent/tool answered this. E.g., 'Weather Worker' or 'Web Search Worker'")

# Internal data structure for the weather specialist
class WeatherStructure(BaseModel):
    temperature: float
    condition: str
    human_response: str

# Base LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# --- 2. DEFINE CUSTOM WORKER TOOLS (THE SUB-AGENTS) ---

def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"it's sunny and 70 degrees in {city}"

def weather_agent_worker(query: str) -> str:
    """
    Useful when you need to answer questions specifically about the weather, climate, or forecasts.
    Input should be the user's raw weather question.
    """
    # This worker binds its own specific tools and schema constraint
    worker_llm_tools = llm.bind_tools([weather_tool])
    worker_structured_llm = llm.with_structured_output(WeatherStructure)
    
    worker_messages = [
        SystemMessage(content="You are a meteorological expert. Use your weather tool to find facts and format them."),
        HumanMessage(content=query)
    ]
    
    # Tool execution loop inside the worker agent
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
    """
    Useful when answering general knowledge, news, current events, or non-weather queries.
    Input should be a search query optimization string.
    """
    ddg_search = DuckDuckGoSearchRun()
    search_result = ddg_search.run(query)
    
    search_messages = [
        SystemMessage(content="You are a research expert. Synthesize raw search results into a clean summary."),
        HumanMessage(content=f"Summarize this finding to answer the user's intent: {search_result}")
    ]
    
    res = llm.invoke(search_messages)
    return f"Search Worker Result -> {res.content}"


# --- 3. SUPERVISOR AGENT CONFIGURATION ---

# Compile workers into a tool array for the Supervisor
agent_tools = [weather_agent_worker, search_agent_worker]

# Bind tools to Supervisor so it can delegate tasks
supervisor_llm_with_tools = llm.bind_tools(agent_tools)
# Force Supervisor to format the definitive final summary response
supervisor_structured_output = llm.with_structured_output(FinalAgentResponse)


# --- 4. EXECUTION FLOW ---

def run_multi_agent_system(user_prompt: str):
    print(f"\nUser Request: '{user_prompt}'")
    
    session_messages = [
        SystemMessage(
            content="You are a team Supervisor. You do not answer questions directly. "
                    "Delegate tasks to your sub-agents (weather_agent_worker or search_agent_worker) "
                    "using tools, collect their findings, and format the absolute final result."
        ),
        HumanMessage(content=user_prompt)
    ]
    
    # Supervisor decides who should handle the input
    supervisor_decision = supervisor_llm_with_tools.invoke(session_messages)
    
    if supervisor_decision.tool_calls:
        session_messages.append(supervisor_decision)
        
        for tool_call in supervisor_decision.tool_calls:
            # Dynamically execute the chosen sub-agent worker function
            chosen_worker = next(t for t in agent_tools if t.__name__ == tool_call["name"])
            print(f"[Supervisor] Routing task to: {chosen_worker.__name__}...")
            
            worker_output = chosen_worker(**tool_call["args"])
            
            # Pass the worker sub-agent's findings back to supervisor history
            session_messages.append(ToolMessage(content=worker_output, tool_call_id=tool_call["id"]))
            
        # Supervisor builds final schema output using the aggregated workers' data
        final_verdict = supervisor_structured_output.invoke(session_messages)
    else:
        # If no sub-agent was needed, compile directly
        final_verdict = supervisor_structured_output.invoke(session_messages)
        
    return final_verdict


# --- 5. VERIFICATION ---
if __name__ == "__main__":
    # Test 1: Triggers Weather Agent
    res1 = run_multi_agent_system("Is it raining in San Francisco right now?")
    print(f"Human Response: {res1.human_response}")
    print(f"Backend Meta: {repr(res1)}\n")
    
    print("-" * 50)

    # Test 2: Triggers Web Search Agent
    res2 = run_multi_agent_system("Who won the latest Formula 1 Monaco Grand Prix?")
    print(f"Human Response: {res2.human_response}")
    print(f"Backend Meta: {repr(res2)}")
