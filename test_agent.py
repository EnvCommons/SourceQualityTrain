"""
Test agent for SourceQualityTrain environment.

Demonstrates how to use the environment with OpenAI's Responses API.
"""

import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/SourceQualityTrain"
    SPLIT = "train"

    # Get API keys from environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable required")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY environment variable required")

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    # Test first few tasks
    for task in tasks[:3]:
        print(f"\n{'='*60}")
        print(f"Task: {task.task_spec['id']}")
        print(f"{'='*60}")

        rollout = or_client.rollout.create(
            run_name="sourcequalitytrain_test",
            rollout_name=f"test_{task.task_spec['id']}",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec
        )

        secrets = {
            "openai_api_key": OPENAI_API_KEY,
            "tavily_api_key": TAVILY_API_KEY
        }

        async with environment.session(task=task, secrets=secrets) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            rollout.log_openai_response(message=input_list[0], is_finished=finished)

            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list
                )

                rollout.log_openai_response(response.output[-1])
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        print(f"\nTool call: {item.name}")
                        args = json.loads(str(item.arguments))
                        print(f"Arguments: {json.dumps(args, indent=2)[:200]}...")

                        tool_result = await session.call_tool(item.name, args)

                        reward = tool_result.reward
                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                        })
                        rollout.log_openai_response(
                            input_list[-1],
                            reward=reward,
                            is_finished=finished
                        )

                        print(f"Reward: {reward:.3f}")

                        if tool_result.finished:
                            finished = True
                            print(f"\n{'='*60}")
                            print("FINISHED!")
                            print(f"Final Reward: {reward}")
                            print(f"{'='*60}")
                            break

        print(f"\nCompleted task: {task.task_spec['id']}")


if __name__ == "__main__":
    asyncio.run(main())
