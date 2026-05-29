import sys
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typing import List, Dict

# Simple tool: get_weather (mock)

def get_weather(location: str) -> str:
    """Mock weather function."""
    return f"The weather in {location} is sunny with 25°C."

# Create memory for checkpointing
memory = MemorySaver()

# Agent configuration
agent = create_agent(
    model="gpt-4o-mini",  # placeholder, replace with actual llm instance if needed
    tools=[get_weather],
    system_prompt="Ты полезный ассистент.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True},
            description_prefix="Подтвердите вызов инструмента",
        ),
    ],
    checkpointer=memory,
)

# Helper to display action requests and collect decisions

def handle_interrupt(interrupt_value: Dict) -> List[Dict]:
    actions = interrupt_value.get("action_requests", [])
    configs = interrupt_value.get("review_configs", {})
    decisions = []
    for idx, act in enumerate(actions):
        name = act.get("name")
        args = act.get("args")
        desc = act.get("description", "")
        print(f"\nAction {idx+1}: {name}")
        print(f"  Args: {args}")
        if desc:
            print(f"  Description: {desc}")
        allowed = configs.get(name, {}).get("allowed_decisions", ["approve", "reject"])
        while True:
            choice = input(f"Choose (a=approve, r=reject{', e=edit' if 'edit' in allowed else ''}): ").strip().lower()
            if choice == "a":
                decisions.append({"type": "approve"})
                break
            elif choice == "r":
                msg = input("Enter rejection reason: ")
                decisions.append({"type": "reject", "message": msg})
                break
            elif choice == "e" and "edit" in allowed:
                new_args = input("Enter edited arguments as JSON (or leave empty to keep): ")
                if new_args:
                    try:
                        import json
                        act["args"] = json.loads(new_args)
                    except Exception:
                        print("Invalid JSON, keeping original.")
                decisions.append({"type": "edit", "new_args": act["args"]})
                break
            else:
                print("Invalid choice. Try again.")
    return decisions

# Main interaction loop
if __name__ == "__main__":
    thread_id = "session-1"
    config = {"configurable": {"thread_id": thread_id}}
    user_msg = input("Введите запрос: ")
    result = agent.invoke({"messages": [{"role": "human", "content": user_msg}]}, config=config)

    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        decisions = handle_interrupt(interrupt_value)
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

    # Final answer
    final_msg = result.get("messages", [])[-1]["content"] if result.get("messages") else ""
    print(f"\nОтвет: {final_msg}")
