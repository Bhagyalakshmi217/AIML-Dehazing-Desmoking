<<<<<<< HEAD
# ClearVision AI — Streamlit Dashboard

A unified local Streamlit interface for:
- NH-HAZE fine-tuned U-Net image dehazing
- SmokeBench U-Net image desmoking
- YOLO person detection/tracking in uploaded video
- PSNR/SSIM evaluation when a matching clean reference is available

## 1. Project structure

```text
ClearVision_AI_Streamlit/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── dehazing_nhhaze_finetuned.pth
    └── desmoke_model.pth
```

Put your trained `.pth` files in the `models` folder. You may also use the sidebar to enter absolute checkpoint paths. The included U-Net architecture matches the classic `ImprovedUNet` checkpoint structure used during development; if `load_state_dict` reports missing/unexpected keys, use the exact model class from your training notebook.

## 2. Copy your checkpoints

From Google Drive, download/copy:
- `AIML_Dehazing_Desmoking/models/dehazing_nhhaze_finetuned.pth`
- `AIML_Dehazing_Desmoking/models/desmoke_model.pth`

Place them in `ClearVision_AI_Streamlit/models/`.

The app defaults to `models/dehazing_nhhaze_finetuned.pth` and `models/desmoke_model.pth`. If your saved SmokeBench checkpoint has a different filename, change the path in the sidebar. Do not rename or overwrite your original checkpoints.

YOLOv8 weights (`yolov8n.pt`) can be downloaded automatically by Ultralytics on first run if internet access is available. Otherwise, download it separately and set the path in the sidebar.

## 3. Install and run (Windows + VS Code)

Open the terminal in this folder:

```bash
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the dashboard:

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 4. Important notes

- The app uses your computer's CPU unless a compatible CUDA-enabled PyTorch installation is configured.
- Restoration runs on 256x256 tiles with 64-pixel overlap, preserving original dimensions. Tiling does not guarantee better restoration; inspect outputs for artifacts.
- The dehazing model was fine-tuned on NH-HAZE. It may not generalize to all types of haze or smoke.
- The desmoking checkpoint should be the SmokeBench model, not the dehazing checkpoint.
- PSNR and SSIM are only meaningful when the uploaded reference is a correctly paired clean ground-truth image. This prototype resizes both images to 256x256 for evaluation.
- Video detections are counted per frame; the total is not the number of unique people.
- This is a research demonstration, not a safety-critical surveillance system.
=======
# AIML-Dehazing-Desmoking
AIML-based image and video dehazing and desmoking with person detection
>>>>>>> origin/main
