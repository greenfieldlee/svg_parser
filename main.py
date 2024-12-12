import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from process_svg import process_svg

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure temporary directories exist
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("temp_outputs", exist_ok=True)

@app.get("/process-svg/")
async def process_svg_info():
    return JSONResponse(content={
        "message": "This endpoint processes SVG files.",
        "usage": "Send a POST request to this endpoint with 'file' (SVG file) and 'grow' (float) in the form data."
    })

@app.post("/process-svg/")
async def process_svg_endpoint(file: UploadFile = File(...), grow: float = Form(0.0)):
    # Generate unique filenames
    temp_svg = f"temp_uploads/{uuid.uuid4()}.svg"
    temp_result_svg = f"temp_outputs/{uuid.uuid4()}.svg"
    temp_png = f"temp_outputs/{uuid.uuid4()}.png"
    
    try:
        # Save uploaded file
        with open(temp_svg, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file to {temp_svg}")
        
        # Process the SVG and get path data
        svg_path_data = process_svg(temp_svg, temp_png, temp_result_svg, grow=grow)
        
        logger.info(f"Generated SVG path data with grow={grow}")
        
        return JSONResponse(content={"svg_path": svg_path_data[1]})
    
    except Exception as e:
        logger.error(f"Error processing SVG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        for temp_file in [temp_svg, temp_png]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        logger.info("Cleaned up temporary files")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

