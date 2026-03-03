import io
import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from local_tester import TextureTester

# Define Pydantic Models for Swagger Documentation
class AnalyzeResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the analysis was successful")
    score: float = Field(..., description="Overall texture score (0-100)")
    v3_score: Optional[float] = Field(None, description="Score from the v3 model (0-100)")
    v5_score: Optional[float] = Field(None, description="Score from the v5 model (0-100)")
    interpretation: str = Field(..., description="Textual interpretation of the score (e.g., 'Smooth skin')")
    color: str = Field(..., description="Hex color code associated with the interpretation")
    original_image: str = Field(..., description="Base64 encoded original input image")
    heatmap_image: str = Field(..., description="Base64 encoded output heatmap image showing texture deviations")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Description of the error that occurred")

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Texture Analysis API",
    description="Microservice for analyzing skin texture using a hybrid v3/v5 intelligent model. Supports face detection, texture map generation, and visual deviation highlighting.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Apply CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the tester model globally on startup (this might take a few seconds)
tester = TextureTester()

def image_to_base64(img_np):
    """Converts an OpenCV image (numpy array) to base64 string"""
    _, buffer = cv2.imencode('.jpg', img_np)
    img_bytes = buffer.tobytes()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"

def check_lighting(img: np.ndarray) -> tuple[bool, str]:
    """
    Checks if the image lighting is within acceptable bounds.
    Returns (is_valid, error_message)
    """
    # Convert to HSV to easily analyze brightness (Value channel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_brightness = np.mean(v_channel)
    
    # Define thresholds
    MIN_BRIGHTNESS = 40.0
    MAX_BRIGHTNESS = 240.0
    
    if mean_brightness < MIN_BRIGHTNESS:
        return False, f"Image is too dark (Brightness: {mean_brightness:.1f}). Please capture in better lighting."
    if mean_brightness > MAX_BRIGHTNESS:
        return False, f"Image is too bright or overexposed (Brightness: {mean_brightness:.1f}). Please reduce lighting."
        
    return True, ""

@app.post(
    "/analyze", 
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid Input or Image Processing Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Analysis"],
    summary="Analyze Skin Texture",
    description="Upload an image containing a face. The API will detect the face, analyze the skin texture, and return a quality score along with a visual heatmap of problem areas."
)
async def analyze_image(file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, etc.)")):
    try:
        # Read the file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "Invalid image file provided."}
            )
            
        # Perform lighting check
        is_lighting_valid, lighting_error = check_lighting(img)
        if not is_lighting_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": lighting_error}
            )
        
        # Create texture map and evaluate
        texture_map, original, error = tester.create_texture_map(img)
        
        if error:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": error}
            )
        
        result = tester.predict(texture_map)
        
        # Parse result (handling both tuple and single float values)
        score = 0.0
        v3_score = None
        v5_score = None
        
        if isinstance(result, tuple) and len(result) == 5:
            score, norm_v3, norm_v5, raw_v3, raw_v5 = result
            v3_score = norm_v3
            v5_score = norm_v5
        elif isinstance(result, tuple):
            score = result[0]
            if len(result) >= 3:
                v3_score = result[1]
                v5_score = result[2]
        else:
            score = result

        interpretation, color = tester.get_interpretation(score)
        color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}" # BGR to RGB hex
        
        # Generate marked heatmap visualization
        deviation = cv2.absdiff(texture_map, 128)
        highlight = cv2.convertScaleAbs(deviation, alpha=8.0)
        
        mask_area = (deviation > 0).astype(np.uint8)
        mask_area = cv2.dilate(mask_area, np.ones((5, 5), np.uint8), iterations=3)
        base_red = mask_area * 40
        
        total_red_add = cv2.add(highlight, base_red)
        
        overlaid_image = original.copy()
        
        # Add to the red channel (BGR format, index 2 is red)
        overlaid_image[:, :, 2] = cv2.add(overlaid_image[:, :, 2], total_red_add)
        
        # Reduce green and blue channels for emphasis
        reduce_val = cv2.convertScaleAbs(highlight, alpha=0.8)
        overlaid_image[:, :, 1] = cv2.subtract(overlaid_image[:, :, 1], reduce_val)
        overlaid_image[:, :, 0] = cv2.subtract(overlaid_image[:, :, 0], reduce_val)
        
        # Convert the generated overlaid image to base64
        overlaid_b64 = image_to_base64(overlaid_image)
        original_b64 = image_to_base64(original)
        
        return {
            "success": True,
            "score": round(score, 1),
            "v3_score": round(v3_score, 1) if v3_score is not None else None,
            "v5_score": round(v5_score, 1) if v5_score is not None else None,
            "interpretation": interpretation,
            "color": color_hex,
            "original_image": original_b64,
            "heatmap_image": overlaid_b64
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

