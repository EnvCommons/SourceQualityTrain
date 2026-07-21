"""Test agent for SourceQualityTrain (terminal-tool style).

The environment uses a hidden @terminal `submit_answer`: agents use
`web_search` and `fetch_url` to find the systematic review and its excluded
studies table, then reply with the verbatim exclusion reason as an ordinary
message. The gpt-5-mini judge grades semantic equivalence.

Runs against the deployed env by default; set LOCAL=1 for localhost:8080.
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
    ENV_NAME = "GeneralReasoning/SourceQualityTrain"
    SPLIT = "train"
    NUM_TASKS = int(os.environ.get("NUM_TASKS", "1"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "40"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable required")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY environment variable required")

    base_url = "http://localhost:8080" if os.environ.get("LOCAL") else None
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)
    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")
    terminal_tool = await environment.terminal_tool()

    print(f"Found {len(tasks)} tasks")
    print(f"Visible tools: {[t['name'] for t in tools]}")
    print(f"Terminal tool (hidden): {terminal_tool}")

    rewards = []
    for task in tasks[:NUM_TASKS]:
        print(f"\n=== Task {task.task_spec['id']} ===")
        print(f"Question: {task.task_spec['question'][:200]}")

        async with environment.session(
            task=task,
            secrets={
                "openai_api_key": OPENAI_API_KEY,
                "tavily_api_key": TAVILY_API_KEY,
            },
        ) as session:
            assistant_ends_rollout = await session.is_assistant_message_final()
            session_tools = await session.list_tools()
            assert "submit_answer" not in [t.name for t in session_tools], \
                "terminal tool leaked into the model's tool list"

            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]

            reward = None
            turn = 0
            while turn < MAX_TURNS:
                turn += 1
                response = await oai_client.responses.create(
                    model=MODEL_NAME, tools=tools, input=input_list,
                )
                input_list += response.output

                calls = [i for i in response.output if i.type == "function_call"]
                if calls:
                    for item in calls:
                        args = json.loads(str(item.arguments))
                        tr = await session.call_tool(item.name, args)
                        text = tr.blocks[0].text if tr.blocks else ""
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": text,
                        })
                        print(f"[{turn}] {item.name}: {json.dumps(args)[:110]}")
                    continue

                final_message = _text_of(response)
                print(f"\n[{turn}] Final message: {final_message[:200]}")

                if not assistant_ends_rollout:
                    print("Not terminal-style; stopping.")
                    break

                out = await session.call_terminal_tool(final_message)
                reward = out.reward
                print(f"call_terminal_tool -> reward={reward} finished={out.finished}")
                print(out.blocks[0].text[:400])
                break

            rewards.append(reward)

    scored = [r for r in rewards if r is not None]
    print(f"\n=== Summary ===")
    print(f"num_tasks={len(rewards)} num_scored={len(scored)} "
          f"mean_reward={sum(scored)/len(scored) if scored else None}")
    print(f"rewards={rewards}")


if __name__ == "__main__":
    asyncio.run(main())
