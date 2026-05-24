import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()

# 1. Initialize the Gemini LLM
# We use gemini-1.5-flash as it is fast and excellent at tool calling
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 2. Define the Tools
# DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# You can easily add more tools to this list later (e.g., Wikipedia, Arxiv, custom functions)
tools = [search_tool]

# 3. Create the Prompt Template with Memory Placeholders
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant equipped with web search capabilities. "
            "Always give accurate answers based on the tools provided when needed.",
        ),
        # This placeholder is where the conversation history will be injected
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        # This placeholder handles the agent's internal scratchpad/thought process
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. Construct the Agent
# We use the tool-calling agent as Gemini natively supports tool/function calling
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # Set to True to see the agent's "thought" process
    handle_parsing_errors=True
)

# 6. Manage Memory / Chat History
# We use a dictionary to store histories so you can manage multiple sessions/users
session_memories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_memories:
        session_memories[session_id] = InMemoryChatMessageHistory()
    return session_memories[session_id]

# Wrap the agent executor with message history capabilities
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- Verification & Execution ---
if __name__ == "__main__":
    config = {"configurable": {"session_id": "user_session_1"}}

    # Turn 1: Asking for real-time information (triggers the search tool)
    print("--- Turn 1 ---")
    response1 = agent_with_chat_history.invoke(
        {"input": str(input("Enter your query: "))},
        config=config
    )
    print(f"\nAI Response: {response1['output']}\n")

    # Turn 2: Testing memory (referencing the previous turn)
    print("--- Turn 2 (Testing Memory) ---")
    response2 = agent_with_chat_history.invoke(
        {"input": "What company did I just ask you about?"},
        config=config
    )
    print(f"\nAI Response: {response2['output']}\n")