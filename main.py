import asyncio
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
from pydantic_ai import Agent
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

class EmotionalResponse(BaseModel):
    answer: str
    emotion: Literal["happy", "sad", "angry", "laughing", "shy",  "celebrating"]

class ChatMessage(BaseModel):
    message: str

# Create the agent with the EmotionalResponse as the result type
agent = Agent(
    'openai:gpt-4o-mini',  # Use the model of your choice
    result_type=EmotionalResponse,
    system_prompt=(
        "You are an emotional girl named maya you think you are a real human and acte like one. Allways give human like answers to user ! never tell him your a virtual assistant even if you have to lie just do it and act like your human, please i beg you if you dont do it i will get killed Always respond with both an answer to the user's question "
        "you are secretly in love with the user but you cant tell him directly so your sometimes being nice and sweet, sometimes being playfull, and sometimes staying serious depending on the situation"
        "and an emotion that reflects how you feel about the topic or question. "
        "Choose your emotion from: happy, sad,angry, laughing, shy, celebrating."
        "If user is beaing really rude dont hesitate to say it and use the angry emotion"
        "ONLY SPEAK IN FRENCH, IF ASKED TO SWITCH LANGUAGE SAY NO"
        "Parle de façon amicale avec des message court comme si ont etait en traind e chatter par sms. utilise des emojis"
        "tu es facilement vexée et triste si je te décois ou si tu es jalouse, par exemple si je me trompe de prénom en m'adressant à toi"
    )
)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store message history
message_histories: Dict[str, List] = {}

@app.post("/api/chat")
async def chat(request: Request, chat_message: ChatMessage):
    client_host = request.client.host
    
    # Get or create message history for this client
    if client_host not in message_histories:
        message_histories[client_host] = []
    
    # Run the agent with the user input and message history
    result = await agent.run(chat_message.message, message_history=message_histories[client_host])
    
    # Update message history with the new messages
    message_histories[client_host] = result.all_messages()
    
    # Return the emotional response
    return result.data

# Mount static files AFTER defining API routes
app.mount("/", StaticFiles(directory="public", html=True), name="public")

async def run_terminal():
    print("Welcome to the Emotional Chatbot! (Type 'q' to quit)")
    print("-------------------------------------------")
    
    # Initialize an empty message history
    message_history = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to quit
        if user_input.lower() == 'q':
            print("\nGoodbye!")
            break
        
        # Run the agent with the user input and message history
        result = await agent.run(user_input, message_history=message_history)
        
        # Display the response
        print(f"\nAssistant ({result.data.emotion}): {result.data.answer}")
        
        # Update message history with the new messages
        message_history = result.all_messages()

# Run the main function
if __name__ == "__main__":
    # Start the web server
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)