import json
import os
from datetime import datetime
import re
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from uuid import uuid4

# Load API key from .env
load_dotenv()
api_key = os.getenv("API_KEY")
model = "open-mixtral-8x7b"

# Initialize Mistral Client
client = MistralClient(api_key=api_key)

# Load preprompt
with open("./preprompt.txt", "r") as f:
    preprompt = f.read()

def get_conversation_filename(suffix=None):
    """Generate or retrieve the filename for the conversation history based on the provided suffix."""
    base_dir = "./convos"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if suffix:
        return os.path.join(base_dir, f"convo_{suffix}.json")
    else:
        # Generate a new unique identifier for the conversation
        new_id = str(uuid4())
        return os.path.join(base_dir, f"convo_{new_id}.json"), new_id

def load_or_initialize_conversation(filename):
    """Load or initialize the conversation history from a file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        return {"model": model, "messages": []}

try:
    conversation_id = None
    while True:
        user_input = str(input("\n>>> "))
        # Check for --convo= suffix in user input
        suffix_match = re.search(r"--convo=(\w+)$", user_input)
        if suffix_match:
            conversation_id = suffix_match.group(1)
            user_input = re.sub(r"\s*--convo=\w+\s*$", "", user_input)  # Clean the command from user input
        
        # Determine the filename based on conversation ID
        if conversation_id:
            filename = get_conversation_filename(conversation_id)
        else:
            filename, conversation_id = get_conversation_filename()
        
        conversation = load_or_initialize_conversation(filename)
        
        # Proceed with adding messages and handling the conversation as before
         # Proceed with adding messages and handling the conversation as before
        conversation['messages'].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate the historical conversation text for the model
        temp_hist = '\n'.join([msg["content"] for msg in conversation['messages'] if msg["role"] == "user"])
        
        messages = [ChatMessage(role="user", content=f"{preprompt}\n\n{temp_hist}\n\n{user_input}")]
        
        # With streaming
        stream_response = client.chat_stream(model=model, messages=messages)

        for chunk in stream_response:
            response_content = chunk.choices[0].delta.content
            print(response_content, end="")
            # Check if the last message was from the model and concatenate if so
            if conversation['messages'] and conversation['messages'][-1]['role'] == 'model':
                conversation['messages'][-1]['content'] += " " + response_content
                conversation['messages'][-1]['timestamp'] = datetime.now().isoformat()  # Update timestamp
                # print(response_content, end=" ")
            else:
                conversation['messages'].append({
                    "role": "model",
                    "content": response_content,
                    "timestamp": datetime.now().isoformat()
                })
            
        # Periodically save the conversation history to ensure no data is lost
        with open(filename, "w") as f:
            json.dump(conversation, f, indent=4)

except KeyboardInterrupt:
    print("Generation Interrupted...")
finally:
    # Save the final state of the conversation to the file
    with open(filename, "w") as f:
        json.dump(conversation, f, indent=4)
    print(f"Stopped Safely and Conversation {conversation_id} History Saved")
