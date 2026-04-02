# PneumoScan — Chest X-Ray Pneumonia Detection
> CNN-powered diagnostic tool · Normal vs Pneumonia classification

---

## Project Structure

```
xray-project/
├── backend/
│   ├── app.py                  # Flask REST API
│   ├── requirements.txt        # Python dependencies
│   └── chest_xray_cnn.keras    # Trained CNN model (7.6 MB)
│
└── frontend/
    └── index.html              # Single-page web UI
```

---

## Model Details

| Property        | Value                          |
|-----------------|--------------------------------|
| Architecture    | Sequential CNN (25 layers)     |
| Input Shape     | 150 × 150 × 3 (RGB)            |
| Output          | Sigmoid (0 = Normal, 1 = Pneumonia) |
| Decision Threshold | 0.5                        |
| Model Size      | 7.6 MB                         |

### Layer Summary
- **Block 1**: Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.25)
- **Block 2**: Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.25)
- **Block 3**: Conv2D(128) × 2 → BatchNorm → MaxPool → Dropout(0.40)
- **Block 4**: Conv2D(256) → BatchNorm → MaxPool → Dropout(0.40)
- **Head**: GlobalAvgPool → Dense(256, ReLU) → BatchNorm → Dropout(0.50) → Dense(1, Sigmoid)

---

## Setup & Run

### 1. Backend (Flask API)

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
# → Running on http://localhost:5000
```

### 2. Frontend

Simply open `frontend/index.html` in your browser.

> ⚠️ If your browser blocks `localhost` requests from a file:// URL,
> serve the frontend with a simple HTTP server:
> ```bash
> cd frontend
> python -m http.server 8080
> # Open http://localhost:8080
> ```

---

## API Reference

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "chest_xray_cnn",
  "input_shape": [150, 150]
}
```

---

### `POST /predict`
Classify a chest X-ray image.

**Option A — multipart/form-data:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@chest_xray.jpg"
```

**Option B — base64 JSON:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,<base64_data>"}'
```

**Response:**
```json
{
  "prediction":  "PNEUMONIA",       // or "NORMAL"
  "confidence":  94.37,             // percent (0–100)
  "raw_score":   0.943721,          // raw sigmoid output
  "threshold":   0.5
}
```

---

## Deploying to Production

### Backend (example with gunicorn)
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### Frontend
Update `API_BASE` in `index.html` to match your server's public URL:
```js
const API_BASE = 'https://your-server.com';
```

Then deploy `index.html` to any static host (Netlify, Vercel, Nginx, S3, etc.).

---

## ⚠️ Disclaimer
This tool is for **research and educational purposes only**.
It is not a medical device and must not be used for clinical diagnosis.
Always consult a qualified radiologist for medical decisions.
