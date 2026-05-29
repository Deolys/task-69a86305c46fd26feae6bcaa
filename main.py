# main.py
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Simple tool: get_weather (mock)

def get_weather(location: str) -> str:
    """Mock weather function returning a fixed string."""
    return f"Сегодня в {location} солнечно, 25°C."

# Create LLM instance (use environment variable for API key)
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

# Memory to preserve state across pauses
memory = MemorySaver()

# Agent with Human-in-the-Loop middleware
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="Ты полезный ассистент, отвечаешь на вопросы и можешь использовать инструменты.",
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
    actions = interrupt_value["action_requests"]
    configs = interrupt_value.get("review_configs", [])
    decisions = []
    for idx, act in enumerate(actions):
        name = act.get("name")
        args = act.get("args")
        desc = act.get("description", "")
        print(f"\n[Action {idx+1}]\nTool: {name}\nArgs: {args}\nDescription: {desc}")
        # Determine allowed decisions
        cfg = next((c for c in configs if c.get("name") == name), {})
        allowed = cfg.get("allowed_decisions", ["approve", "reject", "edit"])
        while True:
            inp = input(f"Выберите действие ({'/'.join(allowed)}): ").strip().lower()
            if inp in {"a": "approve", "r": "reject", "e": "edit"}:
                choice = inp[0]
                if choice == "a":
                    decisions.append({"type": "approve"})
                    break
                elif choice == "r":
                    msg = input("Введите причину отказа: ")
                    decisions.append({"type": "reject", "message": msg})
                    break
                elif choice == "e" and "edit" in allowed:
                    new_args = {}
                    for k, v in args.items():
                        nv = input(f"Изменить {k} (текущее: {v}) -> ") or v
                        new_args[k] = nv
                    decisions.append({"type": "edit", "args": new_args})
                    break
            else:
                print("Неверный ввод. Попробуйте снова.")
    return decisions

# Main interaction loop
if __name__ == "__main__":
    thread_id = os.getenv("THREAD_ID", "session-1")
    config = {"configurable": {"thread_id": thread_id}}
    # Пример запроса пользователя
    user_msg = input("Введите ваш запрос: ")
    result = agent.invoke({"messages": [{"role": "human", "content": user_msg}]}, config=config)
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        decisions = handle_interrupt(interrupt_value)
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
    # Вывод финального ответа
    final_msg = result.get("messages", [])[-1]["content"] if result.get("messages") else "Нет ответа"
    print("\nОтвет ассистента:\n", final_msg)
