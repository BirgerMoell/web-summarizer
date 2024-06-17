# Default
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)


def summarize_using_groq(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You summarize texts that the users sends"
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        model="mixtral-8x7b-32768",
    )

    return chat_completion.choices[0].message.content

print(summarize_using_groq("I am a software engineer."))
