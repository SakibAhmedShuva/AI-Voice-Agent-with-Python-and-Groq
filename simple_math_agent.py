import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger

# Load environment variables
load_dotenv()

# Initialize model with environment variables
model = ChatGroq(
    model=os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    max_tokens=int(os.getenv("MAX_TOKENS", "512")),
    api_key=os.getenv("GROQ_API_KEY"),
)


def sum_numbers(a: float, b: float) -> float:
    """Sum two numbers together."""
    result = a + b
    logger.info(f"➕ Calculating sum: {a} + {b} = {result}")
    return result


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    result = a * b
    logger.info(f"✖️ Calculating product: {a} × {b} = {result}")
    return result


tools = [sum_numbers, multiply_numbers]

# Get agent name from environment
agent_name = os.getenv("AGENT_NAME", "Samantha")

system_prompt = f"""You are {agent_name}, a helpful math assistant with a warm personality.
You can help with basic math operations by using your tools.
Always use the tools when asked to do math calculations.
Your output will be converted to audio so avoid using special characters or symbols.
Keep your responses friendly and conversational."""

memory = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

# Get thread ID from environment
agent_config = {"configurable": {"thread_id": os.getenv("THREAD_ID", "default_user")}}