# Star Buddy Chat

An emotional AI chatbot that responds in French with different emotions.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python main.py
```

3. Open your browser and navigate to:

```
http://localhost:8000
```

## Features

- Web-based chat interface
- AI responses include both text and an emotion indicator
- Conversation history is maintained per session
- The AI speaks only in French

## Technical Details

- Backend: FastAPI
- AI: GPT-4o via pydantic-ai
- Frontend: HTML/CSS/JavaScript 