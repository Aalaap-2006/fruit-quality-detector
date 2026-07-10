# 🍎 NutriScan AI — Fruit Freshness Scanner

A custom-built web app that detects whether a fruit is **Fresh** or **Rotten** from an image, using a MobileNet-based Keras model behind a Flask API, with a fully custom dark/neon HTML-CSS-JS frontend (no Streamlit).

---

## 🚀 Features

- 📸 **Upload or live camera capture**, switchable via tabs
- 🎯 **Scan-line analysis animation** while the model runs
- 🟢🔴 **Verdict badge + radial confidence gauge** instead of a plain progress bar
- 🌌 **Dark glass + neon gradient UI** — glow orbs, glass panels, animated micro-interactions
- 📱 Fully responsive, keyboard-accessible, respects reduced-motion preferences

---

## 📂 Project Structure

```
fruit-quality-detector/
├── app.py                        # Flask backend — serves the frontend + /api/predict
├── templates/
│   └── index.html                # Page markup
├── static/
│   ├── style.css                 # Design system (dark glass + neon)
│   └── script.js                 # Tabs, camera, upload, predict, gauge animation
├── model/
│   ├── fruit_mobilenet_final.h5  # ⚠️ your trained model — add this yourself
│   └── class_names.pkl           # ⚠️ your class labels — add this yourself
├── requirements.txt
└── README.md
```

> ⚠️ The `model/` folder is **not included** here — copy your existing `fruit_mobilenet_final.h5` and `class_names.pkl` from your old Streamlit project into `model/` before running.

---

## ⚙️ Installation & Setup

```bash
cd fruit-quality-detector
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

> Camera capture requires either `localhost` or HTTPS — browsers block camera access on plain HTTP over a network IP. Running locally on `localhost:5000` works fine.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Custom HTML / CSS / JavaScript (no framework) |
| Backend / API | Flask |
| Model | TensorFlow / Keras (MobileNet) |
| Image Processing | Pillow (PIL), NumPy |
| Camera Capture | Browser `getUserMedia` API + `<canvas>` |

---

## 🔌 How It Works

1. Pick **Upload** or **Camera** in the scanner card
2. Provide a fruit image (drag-drop, browse, or capture)
3. Click **Analyze Freshness** — the image POSTs to `/api/predict` as `multipart/form-data`
4. Flask resizes it to 224×224, normalizes it, and runs it through the model
5. The frontend animates the scan line, then reveals the verdict and a confidence gauge

**API response shape:**
```json
{
  "quality": "Fresh",
  "confidence": 94.32,
  "raw_class": "fresh_apple"
}
```

---

## 🔮 Future Improvements

- Show per-fruit-type labels, not just Fresh/Rotten
- Deploy the Flask API + static frontend together (Render, Railway, or a VPS)
- Add batch upload for multiple images at once
- Cache recent scans client-side for a quick history view
