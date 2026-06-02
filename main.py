import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Load environment variables (API keys)
load_dotenv()

# Initialize LLM – replace with your own key or model name
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

# Simple tool: get_weather (mock implementation)
def get_weather(location: str) -> str:
    """Return a mocked weather description for the given location."""
    # In real usage you would call an external API.
    return f"Сегодня в {location} солнечно, 25°C."

# Create agent with Human-in-the-Loop middleware
memory = MemorySaver()
agent = create_agent(
    model=llm,
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
def handle_interrupt(interrupt_value):
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value.get("review_configs", [])
    decisions = []
    for idx, (req, cfg) in enumerate(zip(action_requests, review_configs)):
        name = req.get("name")
        args = req.get("args")
        description = req.get("description", "")
        print(f"\n[Action {idx+1}]\nTool: {name}\nArgs: {args}")
        if description:
            print(f"Description: {description}")
        allowed = cfg.get("allowed_decisions", ["approve", "reject", "edit"])
        # Build prompt for user
        options = []
        if "approve" in allowed:
            options.append("a (approve)")
        if "reject" in allowed:
            options.append("r (reject)")
        if "edit" in allowed:
            options.append("e (edit) – not implemented")
        choice = input(f"Choose action ({', '.join(options)}): ").strip().lower()
        if choice == "a":
            decisions.append({"type": "approve"})
        elif choice == "r":
            msg = input("Enter rejection reason: ")
            decisions.append({"type": "reject", "message": msg})
        else:
            # Default to approve if invalid
            decisions.append({"type": "approve"})
    return decisions

# Main interaction loop
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session-1"}}
    while True:
        user_input = input("\nВведите запрос (или 'exit'): ")
        if user_input.lower() in {"exit", "quit"}:
            break
        # Initial invoke with user message
        result = agent.invoke({"messages": [{"role": "human", "content": user_input}]}, config=config)
        # Process interrupts until completion
        while "__interrupt__" in result:
            interrupt_value = result["__interrupt__"][0].value
            decisions = handle_interrupt(interrupt_value)
            result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
        # Final answer
        final_msg = result.get("messages", [])[-1].content if result.get("messages") else ""
        print(f"\nОтвет: {final_msg}")
