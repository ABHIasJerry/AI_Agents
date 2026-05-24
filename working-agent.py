
import os
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

# 1. Update the schema to include a human-like response field
class WeatherResponse(BaseModel):
    temperature: float = Field(description="The numeric temperature value in Fahrenheit")
    condition: str = Field(description="The weather condition description (e.g., sunny, rainy)")
    human_response: str = Field(description="A warm, natural, human-like conversational response answering the user's question.")

# 2. Tool function definition
def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"it's sunny and 70 degrees in {city}"

# 3. Initialize the model 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,  # Raised slightly from 0 to make the phrasing more natural/creative
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 4. Bind tools to the base LLM first
llm_with_tools = llm.bind_tools([weather_tool])

# 5. Apply the expanded schema constraint second
structured_llm = llm.with_structured_output(WeatherResponse)

# 6. Initialize message session context
messages = [
    SystemMessage(
        content="You are a friendly, conversational assistant. Look up local conditions using tools, "
                "and provide a warm, human-like response alongside the required data structures."
    ),
    HumanMessage(content="What's the weather like in SF right now?")
]

# Step 1: Query the model with tool access
ai_message = llm_with_tools.invoke(messages)

# Step 2: Check if tool calling was triggered
if ai_message.tool_calls:
    messages.append(ai_message)
    
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "weather_tool":
            tool_output = weather_tool(**tool_call["args"])
            messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
    
    # Step 3: Get the final structured output containing both the data and the prose
    final_output = structured_llm.invoke(messages)
else:
    final_output = structured_llm.invoke(messages)

# --- Output the results ---

print("--- Conversational Output (Agent Response) ---")
print(final_output.human_response)

print("\n--- Structured Object (Agent receives from backend) ---")
print(repr(final_output))
