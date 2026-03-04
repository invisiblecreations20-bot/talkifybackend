from fastapi import FastAPI

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Talkify Backend is running successfully!"}
