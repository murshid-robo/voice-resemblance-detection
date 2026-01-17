# api.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
from match_student import match_student

app = FastAPI()

@app.post("/match_reciter")
async def match_reciter(audio: UploadFile = File(...)):
    # Save file temporarily
    path = "temp.wav"
    with open(path, "wb") as f:
        f.write(await audio.read())

    # Call your existing function
    result = match_student(path)

    return {
        "best_reciter": result[0][0],
        "best_score": result[0][1],
        "top_matches": result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
