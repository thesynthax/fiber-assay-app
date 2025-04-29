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

def adjust_brightness_contrast_gamma(img, brightness, contrast, gamma):
    # 1) Brightness & contrast
    buf = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    # 2) Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
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

    ''' # 2) color channel scaling
    b, g, r = cv2.split(img)
    r = np.clip(r.astype(np.float32) * red_boost, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * green_suppress, 0, 255).astype(np.uint8)
    img = cv2.merge([b, g, r])'''

    # b, g, r = cv2.split(img)
    b, g, r = cv2.split(img)

# 1) Identify red‑dominant pixels (simple threshold: red > green)
    red_dom = r > g

# 2) Make float copy, boost red only where red_dom is True
    r_boost = r.astype(np.float32)
    r_boost[red_dom] *= red_boost  # your slider value
    r_boost = np.clip(r_boost, 0, 255).astype(np.uint8)

# 3) (Optional) Suppress green only where green dominates
    green_dom = g > r
    g_supp = g.astype(np.float32)
    g_supp[green_dom] *= green_suppress
    g_supp = np.clip(g_supp, 0, 255).astype(np.uint8)

# 4) Merge back
    img = cv2.merge([b, g_supp, r_boost])

    # 3) noise reduction (bilateral filter)
    filtered = cv2.bilateralFilter(img, noise_reduction, noise_reduction*10, noise_reduction*10)

    # 4) auto‑threshold to create a mask
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5) morphology (only if advanced, else simple opening)
    k = morph_kernel if advanced else 3
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6) remove small artifacts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < artifact_size:
            cv2.drawContours(mask, [c], -1, 0, thickness=cv2.FILLED)

    # 7) apply mask to filtered color image
    cleaned = cv2.bitwise_and(filtered, filtered, mask=mask)
    return cleaned

def analyze_pipeline_color(img: np.ndarray):
    """
    Detect lines on a color image, compute ratios per‑line using an ROI around each
    segment, and overlay red/green pieces back onto the full image.
    """
    # 1) Edge & line detection on full image
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_full, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=10)
    lines = raw if raw is not None else []

    overlay = img.copy()
    ratios = []

    for l in lines:
        x1, y1, x2, y2 = l[0]

        # 2) Compute bounding box + pad by 1px
        pad = 1
        x_min = max(min(x1, x2) - pad, 0)
        x_max = min(max(x1, x2) + pad, img.shape[1] - 1)
        y_min = max(min(y1, y2) - pad, 0)
        y_max = min(max(y1, y2) + pad, img.shape[0] - 1)

        # 3) Extract ROI from color & gray
        roi_color = img[y_min:y_max+1, x_min:x_max+1]
        roi_gray = gray_full[y_min:y_max+1, x_min:x_max+1]

        # 4) Create a mask of the line in ROI‑coords
        line_mask = np.zeros_like(roi_gray)
        pt1 = (x1 - x_min, y1 - y_min)
        pt2 = (x2 - x_min, y2 - y_min)
        cv2.line(line_mask, pt1, pt2, 255, 1)

        ''' # 5) Threshold ROI for red and green
        red_mask_roi = cv2.inRange(roi_color, (0,0,100),    (80,80,255))
        green_mask_roi = cv2.inRange(roi_color, (0,100,0),    (80,255,80))

        # 6) Count pixels of each color only where line_mask==255
        red_len = cv2.countNonZero(cv2.bitwise_and(red_mask_roi,   red_mask_roi,   mask=line_mask))
        green_len = cv2.countNonZero(cv2.bitwise_and(green_mask_roi, green_mask_roi, mask=line_mask))
        total = red_len + green_len
        ratio = red_len / total if total > 0 else 0
        ratios.append(ratio)'''

        # 5) Intensity‐weighted red/green sum along the 1-pixel line mask
        #   split ROI into float channels
        b_roi, g_roi, r_roi = cv2.split(roi_color.astype(np.float32))
        #   extract only the mask pixels
        mask_pixels = line_mask > 0
        sum_r = float(r_roi[mask_pixels].sum())
        sum_g = float(g_roi[mask_pixels].sum())
        total  = sum_r + sum_g
        ratio = sum_r / total if total > 0 else 0.0
        ratios.append(ratio)

        '''# 7) Draw red/green segments on the overlay (full‑image coords)
        #    find the midpoint in image space
        frac = red_len / total if total>0 else 0'''

        # 7) Draw red/green segments on the overlay (full-image coords)
        #    find the split point by the same ratio
        frac = ratio


        mx = int(x1 + (x2 - x1) * frac)
        my = int(y1 + (y2 - y1) * frac)
        cv2.line(overlay, (x1, y1), (mx, my), (0,0,255), 2)
        cv2.line(overlay, (mx, my), (x2, y2), (0,255,0), 2)

    return overlay, ratios

@app.post("/preview")
async def realtime_preview(
    file: UploadFile = File(...),
    brightness: float = Form(0),
    contrast: float = Form(1.0),
    gamma: float = Form(1.0),
    noise_reduction: int = Form(1),
    artifact_size: int = Form(30),
    red_factor: float = Form(1.5),
    green_suppression: float = Form(1.0),
    advanced: bool = Form(False),
    morph_kernel: int = Form(3)
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
    brightness: float = Form(0),
    contrast: float = Form(1.0),
    gamma: float = Form(1.0),
    noise_reduction: int = Form(1),
    artifact_size: int = Form(30),
    red_factor: float = Form(1.5),
    green_suppression: float = Form(1.0),
    advanced: bool = Form(False),
    morph_kernel: int = Form(3)
):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    cleaned = preprocess_color(
        img,
        brightness, contrast, gamma,
        red_factor, green_suppression,
        noise_reduction, artifact_size,
        advanced, morph_kernel
    )
    overlay, ratios = analyze_pipeline_color(cleaned)

    _, buf = cv2.imencode('.jpg', overlay)
    b64 = base64.b64encode(buf).decode('utf-8')
    return JSONResponse({
        "status": "success",
        "data": {
            "lines_detected": len(ratios),
            "ratios": ratios,
            "processed_image": f"data:image/jpeg;base64,{b64}"
        }
    })
