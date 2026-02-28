"""
test_agent.py - Local Testing for OrganicChem1920

Tests the environment with OpenAI's GPT-5.2 model using the Responses API.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python server.py  # In one terminal
    python test_agent.py  # In another terminal
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")


async def main() -> None:
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to local server
    environment = or_client.environments.get(
        name="GeneralReasoning/OrganicChem1920",
        base_url="http://localhost:8080"
    )

    # Get tasks and tools
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} test tasks")

    # Test first task
    task = tasks[0]
    print(f"\n{'='*60}")
    print(f"{'='*60}\n")

    finished = False

    async with environment.session(
        task=task,
        secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        # Get initial prompt
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text if isinstance(prompt, list) else prompt

        print("PROMPT:")
        print(prompt_text)
        print(f"\n{'='*60}\n")

        input_list = [{"role": "user", "content": prompt_text}]

        # Agent loop
        while not finished:
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            # Process model output
            for item in response.output:
                if item.type == "function_call":
                    print(f"TOOL CALL: {item.name}")
                    print(f"Arguments: {item.arguments}\n")

                    # Call tool
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments))
                    )

                    reward = tool_result.reward
                    finished = tool_result.finished

                    # Add tool result to input
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result.blocks[0].text if tool_result.blocks else ""
                    })

                    print(f"REWARD: {reward:.3f}")
                    print(f"\nOUTPUT:")
                    print(tool_result.blocks[0].text if tool_result.blocks else "(no output)")
                    print(f"\n{'='*60}\n")

                    if finished:
                        print("FINISHED!")
                        break

                elif item.type == "text":
                    print(f"MODEL: {item.text}\n")

            # Safety: break if no tool call
            if not any(i.type == "function_call" for i in response.output):
                print("No tool call made, ending session.")
                break


if __name__ == "__main__":
    asyncio.run(main())
