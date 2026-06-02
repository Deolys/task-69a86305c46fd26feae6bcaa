"""
Main script implementing Human-in-the-Loop agent with LangGraph.
The code follows the instructor feedback:
  * Defines `llm` variable.
  * Adds dependency on langgraph.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# LLM definition – replace with your own API key in .env (OPENAI_API_KEY)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Simple tool: get_weather – returns a dummy string for demonstration.
from langchain.tools import Tool

def get_weather(location: str) -> str:
    """Return weather information for the given location."""
    # In a real implementation you would call an API. Here we return a placeholder.
    return f"The weather in {location} is sunny with 25°C."

weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Get current weather for a location. Input: location string. Output: weather description string.",
)

# Agent setup with Human-in-the-Loop middleware
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[weather_tool],
    system_prompt="Ты полезный ассистент.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True},
            description_prefix="Подтвердите вызов инструмента",
        ),
    ],
    checkpointer=memory,
)

# Helper to process interrupt and get decisions from user
def handle_interrupt(interrupt_value):
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value.get("review_configs", [])
    decisions = []
    for idx, req in enumerate(action_requests):
        name = req.get("name")
        args = req.get("args")
        description = req.get("description", "")
        print(f"\n[Action {idx+1}] Tool: {name}")
        print(f"Arguments: {args}")
        if description:
            print(f"Description: {description}")
        # Determine allowed decisions
        config = next((c for c in review_configs if c.get("name") == name), {})
        allowed = config.get("allowed_decisions", ["approve", "reject", "edit"])
        while True:
            choice = input(f"Choose decision (a=approve, r=reject{', e=edit' if 'edit' in allowed else ''}): ").strip().lower()
            if choice == "a":
                decisions.append({"type": "approve"})
                break
            elif choice == "r":
                msg = input("Enter rejection message: ")
                decisions.append({"type": "reject", "message": msg})
                break
            elif choice == "e" and "edit" in allowed:
                new_args = input("Enter new arguments as JSON (or leave empty to keep): ")
                if new_args:
                    try:
                        import json
                        new_args_dict = json.loads(new_args)
                        req["args"] = new_args_dict
                    except Exception as exc:
                        print(f"Invalid JSON: {exc}")
                decisions.append({"type": "edit", "new_args": req.get("args")})
                break
            else:
                print("Invalid choice. Try again.")
    return decisions

# Main interaction loop – single query example
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session-1"}}
    user_input = input("Введите запрос: ")
    result = agent.invoke({"messages": [{"role": "human", "content": user_input}]}, config=config)

    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        decisions = handle_interrupt(interrupt_value)
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

    # Final answer
    final_message = result.get("messages", [])[-1]["content"] if result.get("messages") else ""
    print("\nОтвет агента:")
    print(final_message)
