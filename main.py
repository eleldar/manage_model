from fastapi import FastAPI
from model import get_result

app = FastAPI()


@app.get("/{days}")
async def info(days: int):
    return get_result(days)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
