import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

claude_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
