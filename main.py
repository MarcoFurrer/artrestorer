from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
import uuid
import shutil

app = FastAPI(title="Art Restorer")

# Create necessary directories
UPLOAD_DIR = Path("uploads")
RESTORED_DIR = Path("restored")
UPLOAD_DIR.mkdir(exist_ok=True)
RESTORED_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Mount static directories
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/restored", StaticFiles(directory="restored"), name="restored")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Handle image upload and restoration"""
    try:
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Save uploaded file
        upload_path = UPLOAD_DIR / unique_filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # TODO: Call your AI restoration logic here
        # For now, we'll just copy the file to simulate restoration
        restored_path = RESTORED_DIR / unique_filename
        shutil.copy(upload_path, restored_path)
        
        return JSONResponse({
            "success": True,
            "original_url": f"/uploads/{unique_filename}",
            "restored_url": f"/restored/{unique_filename}",
            "message": "Image processed successfully"
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)