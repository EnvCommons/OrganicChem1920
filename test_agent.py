"""Agent loop for OrganicChem1920.

OrganicChem1920 uses a @terminal tool: the grading tool is hidden from the
model, which answers the chemistry question and writes its response as an
ordinary message. The harness sees a message with no tool calls and routes
its text to session.call_terminal_tool(), which grades it against the
reference answer with gpt-5-mini (continuous score 0.0-1.0).

Runs against the deployed environment by default; set LOCAL=1 to point at a
local `python server.py` on port 8080.
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


def _text_of(response) -> str:
    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return "\n".join(parts).strip()


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "GeneralReasoning/OrganicChem1920"
    SPLIT = os.environ.get("SPLIT", "test")
    NUM_TASKS = int(os.environ.get("NUM_TASKS", "1"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "40"))
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    base_url = "http://localhost:8080" if os.environ.get("LOCAL") else None
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)
    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")
    terminal_tool = await environment.terminal_tool()

    print(f"Found {len(tasks)} tasks")
    print(f"Tools visible to the model: {[t['name'] for t in tools]}")
    print(f"Terminal tool (hidden): {terminal_tool}")

    for task in tasks[:NUM_TASKS]:
        print(f"\n=== Task {task.task_spec['uuid']} ===")

        async with environment.session(
            task=task,
            secrets={"openai_api_key": OPENAI_API_KEY},
        ) as session:
            assistant_ends_rollout = await session.is_assistant_message_final()
            session_tools = await session.list_tools()
            print(f"is_assistant_message_final() -> {assistant_ends_rollout}")
            print(f"session.list_tools() -> {[t.name for t in session_tools]}")
            assert "answer" not in [t.name for t in session_tools], \
                "terminal tool leaked into the model's tool list"

            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]

            reward = None
            turn = 0

            while turn < MAX_TURNS:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                kwargs = {"model": MODEL_NAME, "input": input_list}
                if tools:
                    kwargs["tools"] = tools
                response = await oai_client.responses.create(**kwargs)
                input_list += response.output

                calls = [i for i in response.output if i.type == "function_call"]
                if calls:
                    for item in calls:
                        args = json.loads(str(item.arguments))
                        tool_result = await session.call_tool(item.name, args)
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })
                        print(f"Tool: {item.name}({json.dumps(args)[:120]})")
                    continue

                final_message = _text_of(response)
                print(f"Final message: {final_message[:300]}")

                if not assistant_ends_rollout:
                    print("Environment is not terminal-tool style; stopping.")
                    break

                out = await session.call_terminal_tool(final_message)
                reward = out.reward
                print(f"\ncall_terminal_tool -> reward={reward} finished={out.finished}")
                print(out.blocks[0].text[:800])
                break

            print(f"\n=== Task {task.task_spec['uuid']} reward={reward} turns={turn} ===")


if __name__ == "__main__":
    asyncio.run(main())
