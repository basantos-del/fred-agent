from tools import get_portfolio_context
import json

context = get_portfolio_context()
print(json.dumps(context, indent=2))
