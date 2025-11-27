from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
import uuid
import shutil
import base64

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
    """Handle image upload"""
    try:
        # Generate unique filename with simple naming
        file_ext = os.path.splitext(file.filename)[1]
        # Use simple naming like image1.png
        base_name = f"image{uuid.uuid4().hex[:8]}"
        filename = f"{base_name}{file_ext}"
        
        # Save uploaded file
        upload_path = UPLOAD_DIR / filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "base_name": base_name,
            "url": f"/uploads/{filename}",
            "message": "Image uploaded successfully"
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error uploading image: {str(e)}"
        }, status_code=500)


@app.post("/process")
async def process_with_mask(filename: str = Form(...), mask_data: str = Form(...)):
    """Handle mask and process restoration"""
    try:
        # Parse base name from filename
        base_name = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1]
        
        # Save mask with naming convention: image1_mask001.png
        mask_filename = f"{base_name}_mask001.png"
        mask_path = UPLOAD_DIR / mask_filename
        
        # Decode base64 mask data
        mask_data = mask_data.split(',')[1]  # Remove data:image/png;base64,
        mask_bytes = base64.b64decode(mask_data)
        
        # Save mask as PNG
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)
        
        # TODO: Call your AI restoration logic here with image and mask
        # For now, we'll just copy the file to simulate restoration
        upload_path = UPLOAD_DIR / filename
        restored_filename = f"{base_name}_restored{file_ext}"
        restored_path = RESTORED_DIR / restored_filename
        shutil.copy(upload_path, restored_path)
        
        return JSONResponse({
            "success": True,
            "original_url": f"/uploads/{filename}",
            "mask_url": f"/uploads/{mask_filename}",
            "restored_url": f"/restored/{restored_filename}",
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