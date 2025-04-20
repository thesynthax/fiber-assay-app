## backend/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import base64
from io import BytesIO

app = FastAPI()
# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/convert")
async def convert_tiff(file: UploadFile = File(...)):
    # read the raw bytes
    data = await file.read()
    np_img = np.frombuffer(data, np.uint8)
    # decode TIFF (OpenCV supports TIFF)
    img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
    # encode to PNG
    success, buf = cv2.imencode(".png", img)
    if not success:
        return JSONResponse({"error": "Could not convert image"}, status_code=500)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/png")

'''
def enhance_red(img: np.ndarray, red_factor: float = 1.5) -> np.ndarray:
    """ Boost red channel before any denoising """
    b, g, r = cv2.split(img)
    r = np.clip(r.astype(np.float32) * red_factor, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def preprocess(
    img: np.ndarray,
    threshold_val: int = 45,
    bilateral_d: int = 9,
    bilateral_sigmaColor: int = 75,
    bilateral_sigmaSpace: int = 75,
    morph_kernel: int = 3,
) -> np.ndarray:
    """
    1) Grayscale threshold mask
    2) Color bilateral filter
    3) Remove yellow overlap
    4) Morphological cleanup
    5) Return color image with noise suppressed
    """
    # 1. mask from gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    # 2. bilateral on color
    filtered = cv2.bilateralFilter(
        img, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace
    )

    # 3. remove yellow
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (20,100,100), (30,255,255))
    inv_yellow = cv2.bitwise_not(yellow)
    combined_mask = cv2.bitwise_and(mask, inv_yellow)

    # 4. morphology
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    clean = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel)
    clean = cv2.morphologyEx(clean,           cv2.MORPH_CLOSE, kernel)

    # 5. apply to color
    return cv2.bitwise_and(filtered, filtered, mask=clean)

def filter_duplicate_lines(lines, distance_thresh=50, angle_thresh=10):
    """Merge or remove near-duplicate lines based on proximity & orientation"""
    def angle(line):
        x1, y1, x2, y2 = line[0]
        return abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
    filtered = []
    for l1 in lines:
        x1,y1,x2,y2 = l1[0]
        ang1 = angle(l1)
        keep = True
        for l2 in filtered:
            x3,y3,x4,y4 = l2[0]
            ang2 = angle(l2)
            d1 = np.hypot(x1-x3, y1-y3)
            d2 = np.hypot(x2-x4, y2-y4)
            if d1 < distance_thresh and d2 < distance_thresh and abs(ang1-ang2) < angle_thresh:
                # keep only longer
                if np.hypot(x2-x1,y2-y1) > np.hypot(x4-x3,y4-y3):
                    filtered.remove(l2)
                    filtered.append(l1)
                keep = False
                break
        if keep:
            filtered.append(l1)
    return filtered


def analyze_pipeline(img: np.ndarray, red_factor: float = 1.5):
    """
    Detect lines on a color image, compute ratios, and overlay segments.
    """
    # grayscale for edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=30, maxLineGap=10)
    lines = raw if raw is not None else []

    # masks for measuring
    red_mask   = cv2.inRange(img, (0,0,100),    (80,80,255))
    green_mask = cv2.inRange(img, (0,100,0),    (80,255,80))

    overlay = img.copy()
    ratios = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        lm = np.zeros_like(gray)
        cv2.line(lm, (x1,y1),(x2,y2), 255, 1)
        rl = cv2.countNonZero(cv2.bitwise_and(red_mask,   red_mask,   mask=lm))
        gl = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=lm))
        tot = rl + gl
        ratio = rl/tot if tot>0 else 0
        ratios.append(ratio)
        # draw
        mx = int(x1 + (x2 - x1)*(rl/tot)) if tot>0 else x1
        my = int(y1 + (y2 - y1)*(rl/tot)) if tot>0 else y1
        cv2.line(overlay, (x1,y1),(mx, my), (0,0,255), 2)
        cv2.line(overlay, (mx, my),(x2,y2), (0,255,0), 2)

    return overlay, ratios

@app.post("/preview")
async def realtime_preview(
    file: UploadFile = File(...),
    threshold: int = Form(45),
    bilateral_d: int = Form(9),
    morph_k: int = Form(3),
    red_factor: float = Form(1.5)
):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 1) Enhance red first
    img_enh = enhance_red(img, red_factor)
    # 2) Then denoise/color‐preserve
    preview = preprocess(
        img_enh,
        threshold_val=threshold,
        bilateral_d=bilateral_d,
        bilateral_sigmaColor=75,
        bilateral_sigmaSpace=75,
        morph_kernel=morph_k
    )

    _, buf = cv2.imencode('.jpg', preview)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type='image/jpeg')
   
@app.post("/upload")
async def analyze(
    file: UploadFile = File(...),
    threshold: int = Form(45),
    bilateral_d: int = Form(9),
    morph_k: int = Form(3),
    red_factor: float = Form(1.5)
):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # red enhance → user preprocess → final analysis
    img_enh  = enhance_red(img, red_factor)
    cleaned  = preprocess(img_enh, threshold, bilateral_d,75,75,morph_k)
    overlay, ratios = analyze_pipeline(cleaned)

    _, buf = cv2.imencode('.jpg', overlay)
    b64    = base64.b64encode(buf).decode('utf-8')
    return JSONResponse({
      "status":"success",
      "data":{
        "lines_detected": len(ratios),
        "ratios": ratios,
        "processed_image": f"data:image/jpeg;base64,{b64}"
      }
    })
    '''

def adjust_brightness_contrast_gamma(img, brightness, contrast, gamma):
    # 1) Brightness & contrast
    buf = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    # 2) Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(buf, table)

def preprocess_color(
    img: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
    red_boost: float,
    green_suppress: float,
    noise_reduction: int,
    artifact_size: int,
    advanced: bool,
    morph_kernel: int
) -> np.ndarray:
    # 1) brightness/contrast/gamma
    img = adjust_brightness_contrast_gamma(img, brightness, contrast, gamma)

    # 2) color channel scaling
    b, g, r = cv2.split(img)
    r = np.clip(r.astype(np.float32) * red_boost,      0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * green_suppress, 0, 255).astype(np.uint8)
    img = cv2.merge([b, g, r])

    # 3) noise reduction (bilateral filter)
    filtered = cv2.bilateralFilter(img,
                                   noise_reduction,
                                   noise_reduction*10,
                                   noise_reduction*10)

    # 4) auto‑threshold to create a mask
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5) morphology (only if advanced, else simple opening)
    k = morph_kernel if advanced else 3
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6) remove small artifacts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < artifact_size:
            cv2.drawContours(mask, [c], -1, 0, thickness=cv2.FILLED)

    # 7) apply mask to filtered color image
    cleaned = cv2.bitwise_and(filtered, filtered, mask=mask)
    return cleaned

def analyze_pipeline_color(img: np.ndarray):
    # edge & line detection
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw   = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=20, minLineLength=30, maxLineGap=10)
    lines = raw if raw is not None else []

    # prepare masks for ratio
    red_mask   = cv2.inRange(img, (0,0,100),    (80,80,255))
    green_mask = cv2.inRange(img, (0,100,0),    (80,255,80))

    overlay = img.copy()
    ratios  = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        lm = np.zeros_like(gray)
        cv2.line(lm, (x1,y1),(x2,y2), 255, 1)
        rl = cv2.countNonZero(cv2.bitwise_and(red_mask,   red_mask,   mask=lm))
        gl = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=lm))
        tot = rl + gl
        ratio = rl/tot if tot>0 else 0
        ratios.append(ratio)
        # draw red/green segments
        mx = int(x1 + (x2-x1)*(rl/tot)) if tot>0 else x1
        my = int(y1 + (y2-y1)*(rl/tot)) if tot>0 else y1
        cv2.line(overlay, (x1,y1),(mx, my), (0,0,255), 2)
        cv2.line(overlay, (mx, my),(x2,y2), (0,255,0), 2)

    return overlay, ratios

@app.post("/preview")
async def realtime_preview(
    file: UploadFile = File(...),
    brightness:    float = Form(0),
    contrast:      float = Form(1.0),
    gamma:         float = Form(1.0),
    noise_reduction: int = Form(1),
    artifact_size:  int = Form(30),
    red_factor:     float = Form(1.5),
    green_suppression: float = Form(1.0),
    advanced:       bool  = Form(False),
    morph_kernel:   int   = Form(3)
):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    cleaned = preprocess_color(
        img,
        brightness, contrast, gamma,
        red_factor, green_suppression,
        noise_reduction, artifact_size,
        advanced, morph_kernel
    )
    _, buf = cv2.imencode('.jpg', cleaned)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type='image/jpeg')

@app.post("/upload")
async def analyze(
    file: UploadFile = File(...),
    brightness:    float = Form(0),
    contrast:      float = Form(1.0),
    gamma:         float = Form(1.0),
    noise_reduction: int = Form(1),
    artifact_size:  int = Form(30),
    red_factor:     float = Form(1.5),
    green_suppression: float = Form(1.0),
    advanced:       bool  = Form(False),
    morph_kernel:   int   = Form(3)
):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    cleaned = preprocess_color(
        img,
        brightness, contrast, gamma,
        red_factor, green_suppression,
        noise_reduction, artifact_size,
        advanced, morph_kernel
    )
    overlay, ratios = analyze_pipeline_color(cleaned)

    _, buf = cv2.imencode('.jpg', overlay)
    b64    = base64.b64encode(buf).decode('utf-8')
    return JSONResponse({
        "status": "success",
        "data": {
            "lines_detected": len(ratios),
            "ratios": ratios,
            "processed_image": f"data:image/jpeg;base64,{b64}"
        }
    })
