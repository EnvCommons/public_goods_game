import asyncio
import json
import os
from openai import AsyncOpenAI
from openreward import OpenReward

async def test_agent():
    or_client = OpenReward()
    oai_client = AsyncOpenAI()
    MODEL = "gpt-5.2"
    env = or_client.environments.get(name="PublicGoodsGameEnvironment", base_url="http://localhost:8080")
    tasks = await env.list_tasks(split="test")
    tools = await env.list_tools(format="openai")
    print(f"Tasks: {len(tasks)}, Tools: {[t['function']['name'] for t in tools]}")
    task = tasks[0]
    async with env.session(task=task, secrets={"openai_api_key": os.getenv("OPENAI_API_KEY")}) as session:
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]
        finished = False
        turns = 0
        while not finished and turns < 100:
            response = await oai_client.responses.create(model=MODEL, tools=tools, input=input_list)
            input_list += response.output
            for item in response.output:
                if item.type == "function_call":
                    result = await session.call_tool(item.name, json.loads(str(item.arguments)))
                    finished = result.finished
                    input_list.append({"type": "function_call_output", "call_id": item.call_id, "output": result.blocks[0].text})
                    turns += 1
                    print(f"Turn {turns}: {item.name}({item.arguments}) -> reward={result.reward}, finished={finished}")
                    if finished:
                        break
    print(f"\nAgent test complete after {turns} turns")

if __name__ == "__main__":
    asyncio.run(test_agent())
