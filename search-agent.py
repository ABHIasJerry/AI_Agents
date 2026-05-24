# IMPORTS
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# LOAD ENVIRONMENT VARIABLES
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")  # Gemini API key

# # LLM SETUP
# llm = OpenAI(temperature=0, openai_api_key=key)

# LLM SETUP (Gemini instead of OpenAI)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # you can also try "gemini-1.5-pro" or "gemini-1.5-flash" or "gemini-pro"
    temperature=0,
    google_api_key=google_api_key
)

# MEMORY SETUP (remembers past conversation turns)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# TOOLS SETUP
search = DuckDuckGoSearchRun()
wiki = WikipediaAPIWrapper()
google_search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for answering questions about current events or general web queries."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for factual and historical information."
    ),
    Tool(
        name="Google Search",
        func=google_search.run,
        description="Useful for broader search queries with more detailed results."
    )
]

# CUSTOMIZE AGENT BEHAVIOR
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # conversational agent
    memory=memory,  # adds memory
    verbose=True,
    handle_parsing_errors=True  # makes agent more robust
)

# RUN AGENT LOOP
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    response = agent.run(query)
    print("Agent Response: -> ", response)
