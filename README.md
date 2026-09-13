# Cellscope — Blood Cell Detection (Full Stack)

A full-stack app around your YOLOv8 blood-cell detection model: upload a microscopy
image, get an annotated result plus automatic RBC/WBC/Platelet counts and ratios,
and export the detections as CSV.

```
fullstack/
├── backend/
│   ├── main.py            # FastAPI inference API
│   ├── requirements.txt
│   └── models/
│       └── best.pt        # <-- put YOUR trained weights here
├── frontend/
│   └── index.html         # single-file interactive UI
└── README.md
```

## 1. Get your trained weights

From the Lab 6 Colab notebook, after training:

```python
best_weights = model.trainer.best
```

Download that `best.pt` file and place it at `backend/models/best.pt`.

## 2. Run the backend locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it's alive: open http://localhost:8000/health — should show `"model_loaded": true`.

## 3. Run the frontend locally

Just open `frontend/index.html` directly in your browser (double-click it),
or serve it with any static server, e.g.:

```bash
cd frontend
python -m http.server 5500
```

Then visit http://localhost:5500. The "API endpoint" field at the top defaults
to `http://localhost:8000` — leave it as is for local use.

## 4. Deploy for free

**Backend → Hugging Face Spaces (recommended, free CPU):**
1. Create a new Space → SDK: "Docker" (or "Gradio"/"FastAPI" template if offered).
2. Push `backend/` contents (add a small `Dockerfile` that runs
   `uvicorn main:app --host 0.0.0.0 --port 7860`, since Spaces expects port 7860).
3. Commit `best.pt` to the Space (use Git LFS if it's over ~10MB, though a
   YOLOv8n weight file is usually only a few MB).

**Backend → Render (alternative):**
1. New Web Service → connect your GitHub repo → root directory `backend/`.
2. Build command: `pip install -r requirements.txt`
   Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Free tier spins down when idle — first request after inactivity is slow.

**Frontend → Vercel or Netlify:**
1. Push `frontend/` to GitHub, import the repo, deploy as a static site.
2. After deploying, open your live site and update the "API endpoint" field
   to your deployed backend URL (e.g. `https://your-space.hf.space`).

## Notes

- This model is trained on a small (364-image) public dataset for a university
  lab exercise. It is a learning/demo project, **not** a validated diagnostic
  tool — the UI includes a disclaimer to that effect, keep it.
- CORS is wide open (`allow_origins=["*"]`) for ease of local development.
  Restrict it to your actual frontend domain before sharing the deployed link
  widely.
