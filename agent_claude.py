import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from tools import tool_functions, tools, to_anthropic_tools

load_dotenv()
claude_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
claude_tools = to_anthropic_tools(tools)


def run_agent_turn_claude(user_message, system_prompt):
    messages = [{"role": "user", "content": user_message}]
    used_tools = []

    while True:
        response = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system=system_prompt,
            messages=messages,
            tools=claude_tools
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    used_tools.append(block.name)
                    function_to_call = tool_functions[block.name]

                    try:
                        result = function_to_call(**block.input)
                    except Exception as e:
                        result = {"error": f"Tool '{block.name}' failed: {e}"}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text, used_tools
