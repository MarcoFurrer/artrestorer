# 🎨 AI Art Restorer

Restore damaged or aged artwork using AI-powered image enhancement, inpainting, and style-preserving reconstruction.

## Features

- 📤 Drag & drop or click to upload images
- 🎨 AI-powered art restoration

## Prerequisites

- Python 3.13 or higher
- pip or uv package manager

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd artrestorer
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Running the Application

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

The application will be available at:
- **Local**: http://localhost:8000
- **Network**: http://0.0.0.0:8000