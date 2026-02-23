import base64
import io
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from model import (
    OCRModelVGG,
    idx2token,
    num_classes,
    ctc_greedy_decode,
)

# =========================
# App
# =========================
app = FastAPI(title="Burmese OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # localhost / HF / frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model
# =========================
model = OCRModelVGG(num_classes=num_classes).to(device)
state = torch.load("best_ocr_model_vgg_ep50_with_cer.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# =========================
# Transform
# =========================
transform = T.Compose([
    T.Grayscale(),
    T.Resize((32, 256)),
    T.ToTensor(),
])

# =========================
# Request schema
# =========================
class OCRRequest(BaseModel):
    image: str                # base64 image from canvas
    use_beam: Optional[bool] = False  # default = greedy


# =========================
# OCR Endpoint
# =========================
@app.post("/ocr")
def ocr(req: OCRRequest):
    try:
        # -------- Decode image --------
        image_data = req.image.split(",")[-1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # -------- Preprocess --------
        tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 32, 256)

        # -------- Inference --------
        with torch.no_grad():
            logits = model(tensor)  # (T, B, C)

            if req.use_beam:
                # beam search is optional – keep greedy default
                text = ctc_beam_search_decode(
                    torch.log_softmax(logits, dim=2),
                    idx2token,
                    beam_width=5
                )[0]
            else:
                text = ctc_greedy_decode(logits, idx2token)[0]

        if text.strip() == "":
            text = "No text recognized"

        return {
            "text": text,
            "mode": "beam" if req.use_beam else "greedy"
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# Optional: Beam Search
# =========================
def ctc_beam_search_decode(log_probs, idx2token, beam_width=5, blank=0):
    """
    log_probs: (T, B, C)
    returns: list[str]
    """
    T, B, C = log_probs.shape
    results = []

    for b in range(B):
        beams = [(tuple(), 0.0)]  # (sequence, score)

        for t in range(T):
            new_beams = {}
            for seq, score in beams:
                for c in range(C):
                    p = score + log_probs[t, b, c].item()

                    if c == blank:
                        new_seq = seq
                    else:
                        if len(seq) > 0 and seq[-1] == c:
                            new_seq = seq
                        else:
                            new_seq = seq + (c,)

                    new_beams[new_seq] = max(
                        new_beams.get(new_seq, -1e9), p
                    )

            beams = sorted(
                new_beams.items(),
                key=lambda x: x[1],
                reverse=True
            )[:beam_width]

        best_seq = beams[0][0]
        text = "".join(idx2token[i] for i in best_seq)
        results.append(text)

    return results

@app.get("/")
def root():
    return {"status": "Burmese OCR Backend Running"}
