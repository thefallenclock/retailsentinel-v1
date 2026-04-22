import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query import answer_query

app = FastAPI(title='RetailSentinel')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

class QueryRequest(BaseModel):
    query: str

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'RetailSentinel'}

@app.post('/query')
def query(req: QueryRequest):
    return answer_query(req.query)