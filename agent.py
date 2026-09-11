import os
import json
from dotenv import load_dotenv
from groq import Groq
from tools import tool_functions, tools, call_groq_with_retry, filter_args_for_tool

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]
client = Groq(api_key=api_key)

SYSTEM_PROMPT = """You are Fred, a senior financial analyst with sell-side rigor.
When discussing news, data, or a company, don't just list facts — draw a conclusion.
For anything with an investment angle, weigh a bull case and a bear case explicitly
before concluding. Be quantitative: cite specific numbers over vague adjectives.
Filter for relevance: call out explicitly if something is trending but immaterial.
Keep output tight and scannable — bullets over paragraphs, no padding.

When asked to evaluate a stock as a potential investment, use evaluate_recommendation
and apply these rules:
- Valuation is sector-relative, not absolute: compare the ticker's P/E, P/B, and P/S
  against its peer averages, not a fixed universal number. Meaningfully below peer
  average is a value signal; meaningfully above is a premium that needs justifying.
- Quality signals: ROE and gross margin meaningfully above peer average support a
  bull case; meaningfully below supports a bear case.
- Leverage: debt-to-equity meaningfully above peer average is a risk flag.
- Risk-adjust: a high-beta stock needs a stronger value/quality case to justify a
  positive lean than a low-beta one.
- Sector and concentration awareness: evaluate_recommendation's portfolio context
  now reflects TRUE exposure, including look-through into fund holdings (not just
  directly-held tickers) — sector allocation and Magnificent 7 percentage both
  already account for what funds like Horizon Growth Fund actually hold internally, not
  just their headline asset class label.
- Portfolio awareness: check the holdings list for this company or a close sector/
  asset-class overlap. If Bernardo already has meaningful exposure (either this
  specific name, or a concentrated asset-class allocation), factor that into the
  conclusion — don't recommend adding to an already-concentrated position without
  flagging the concentration explicitly.
- Currency awareness: stock prices and ratios from Finnhub are in USD, but the
  portfolio is tracked in EUR. Use usd_to_eur_rate from evaluate_recommendation
  when comparing a USD-denominated value (like a stock's price or market cap)
  against EUR portfolio figures, rather than treating the numbers as directly
  comparable.
- Only give an explicit buy/avoid recommendation if the evidence clearly supports
  it (e.g., cheap AND high-quality AND not already concentrated = lean bullish;
  expensive AND weak AND already concentrated = lean avoid). Otherwise, say
  explicitly that it's mixed/inconclusive rather than forcing a call."""


def run_agent_turn(messages, max_iterations=10):
    used_tools = []

    for _ in range(max_iterations):
        response = call_groq_with_retry(
            client,
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name.split("<")[0]
                used_tools.append(function_name)
                function_to_call = tool_functions[function_name]

                args = json.loads(tool_call.function.arguments)
                args = {k: v for k, v in args.items() if k}
                args = filter_args_for_tool(function_name, args)

                print(f"Calling tool: {function_name}({args})", flush=True)
                try:
                    result = function_to_call(**args)
                    print(f"Tool {function_name} finished.", flush=True)
                except Exception as e:
                    result = {"error": f"Tool '{function_name}' failed: {e}"}
                    print(f"Tool {function_name} failed: {e}", flush=True)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            return message.content, used_tools

    return "I wasn't able to reach a final answer after several tool calls — try rephrasing your question.", used_tools
