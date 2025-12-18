from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
import uuid
import shutil
import base64
import httpx
from io import BytesIO
import asyncio

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
    try:
        base_name = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1]

        # Maske speichern
        mask_filename = f"{base_name}_mask001.png"
        mask_path = UPLOAD_DIR / mask_filename
        mask_data_str = mask_data.split(',')[1]
        mask_bytes = base64.b64decode(mask_data_str)
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)

        # Original Bild laden
        upload_path = UPLOAD_DIR / filename
        with open(upload_path, "rb") as f:
            image_bytes = f.read()

        # URLs für beide Modelle
        urls = {
            "normal": "https://lama-restorer-api-210237704517.europe-west4.run.app/inpaint",
            "finetuned": "https://lama-restorer2-api-210237704517.europe-west4.run.app/inpaint"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Parallel abschicken
            tasks = [
                client.post(urls["normal"], files={'image': (filename, BytesIO(image_bytes)),
                                                   'mask': (mask_filename, BytesIO(mask_bytes))}),
                client.post(urls["finetuned"], files={'image': (filename, BytesIO(image_bytes)),
                                                      'mask': (mask_filename, BytesIO(mask_bytes))})
            ]
            responses = await asyncio.gather(*tasks)

        results = {}
        for key, resp in zip(["normal", "finetuned"], responses):
            if resp.status_code == 200:
                res_filename = f"{base_name}_{key}{file_ext}"
                with open(RESTORED_DIR / res_filename, "wb") as f:
                    f.write(resp.content)
                results[f"{key}_url"] = f"/restored/{res_filename}"
            else:
                results[f"{key}_url"] = None

        return JSONResponse({
            "success": True,
            "original_url": f"/uploads/{filename}",
            "results": results
        })

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)