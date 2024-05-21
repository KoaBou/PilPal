import dotenv
import asyncio
import uvicorn

from chatbot import Chatbot

from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

dotenv.load_dotenv()
app = FastAPI()

patient_info = {
    'name': 'Nguyen Trung Nguyen',
    'gender': 'Male',
    'medical record': 'Blood Clots'
}

chatbot = Chatbot(patient_info)


class Message(BaseModel):
    query: str


async def run_call(query: str, stream_it: AsyncIteratorCallbackHandler):
    chatbot.chain.llm.callbacks.append(stream_it)
    response = await chatbot.chain.ainvoke(query)

    return response


async def create_gen(query: str, stream_it: AsyncIteratorCallbackHandler):
    task = asyncio.create_task(run_call(query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task


@app.get("/chat")
async def chat(
        query: Message = Body(...),
):
    stream_it = AsyncIteratorCallbackHandler()
    gen = create_gen(query.query, stream_it)

    return StreamingResponse(gen, media_type="text/event-stream")




@app.get("/health")
async def health():
    return {"status": "I'm fine!"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True
    )