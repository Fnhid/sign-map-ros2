| Warning: this repo is in work, not tested so very unstable


# Sign-to-PointCloud Web Pipeline

This folder builds the workflow:

1. VGGT-SLAM 2.0 creates framewise pointcloud logs from an image sequence.
2. Fine-tuned DETR detects `sign` in each frame.
3. OCR extracts text from each detected sign crop.
4. 2D detections are mapped to 3D world points.
5. A web viewer (`viewer.html`) shows point cloud + detected sign text markers.

## Files

- `run_full_pipeline.sh`: end-to-end runner (SLAM + DETR/OCR/3D + web view)
- `pipeline.py`: stage-2 processor (uses existing framewise pointcloud logs)

## Conda Setup

This pipeline uses two Python environments by default:

- `vggt-slam`: for VGGT-SLAM 2.0
- `sign-map-ros2`: for stage-2 detection/OCR/3D mapping

The commands below use the same environment names that this repo already expects in `run_full_pipeline.sh`.

### 1) VGGT-SLAM 2.0 environment

Following the upstream `VGGT-SLAM` setup in this workspace:

```bash
cd /workspace/VGGT-SLAM
conda create -n vggt-slam python=3.11
conda activate vggt-slam
chmod +x setup.sh
./setup.sh
```

This repo expects:

- `VGGT_REPO=./VGGT-SLAM`
- `VGGT_PY_REPO=./vggt`

relative to this repository root unless overridden with environment variables.

### 2) Stage-2 environment

Example setup for the environment used by `pipeline.py`:

```bash
conda create -n sign-map-ros2 python=3.11 -y
conda activate sign-map-ros2
pip install numpy opencv-python scipy
```

Depending on your OCR / detector choice, you may also need extra packages such as EasyOCR, Tesseract bindings, or other model-specific dependencies already used in your local workspace.

## Quick Start

```bash
chmod +x ./run_full_pipeline.sh
./run_full_pipeline.sh /path/to/image_sequence_folder
```

Video input is also supported:

```bash
./run_full_pipeline.sh /path/to/video.mp4
```

Optional 3rd argument sets frame extraction FPS for video:

```bash
./run_full_pipeline.sh /path/to/video.mp4 ./runs/my_run 10
```

Useful env knobs for long videos:

```bash
EXTRACT_MAX_SECONDS=30 STAGE2_MAX_FRAMES=200 ./run_full_pipeline.sh /path/to/video.mp4
```

- `EXTRACT_MAX_SECONDS`: extract only first N seconds from video
- `EXTRACT_START_SECONDS`: start extracting from N seconds
- `STAGE2_MAX_FRAMES`: max frames used in DETR/OCR stage (`0` = all)
- `MAX_LOOPS`: loop closure setting for VGGT-SLAM (`0` default in this script)
- `RUN_OS=1`: enable VGGT-SLAM open-set mode (`--run_os`, interactive text-query + SAM3 3D OBB in viser)
- `OS_QUERIES="door,sign"`: non-interactive open-set queries to run and save as JSON
- `OS_SAVE_PATH=/path/to/open_set_3d.json`: output JSON consumed by stage-2 viewer overlay
- `SAM3_CKPT_PATH=/path/to/sam3.pt`: use local SAM3 checkpoint file directly
- `SAM3_LOAD_FROM_HF=0`: disable HF download path and require local checkpoint
- `DETECTOR_BACKEND=sam3|hybrid|detr`: sign detector source in stage-2 (`sam3` default in `run_full_pipeline.sh`)
- `SAM3_QUERY_FILTER=sign`: which open-set query labels to use as sign boxes
- `SAM3_FALLBACK_TO_DETR=1`: when `DETECTOR_BACKEND=sam3`, fallback to DETR if a frame has no SAM3 sign box
- `OCR_NUMERIC_ROI_STRICT=0`: if numeric ROI fails, fallback to full-crop OCR (recommended)
- `OCR_NUMERIC_CHARS="B0123456789 "`: OCR allowlist (basement `B` + digits)
- `OCR_TEXT_PATTERN='[^B0-9\\s]+'`: strip everything except `B`, digits, and whitespace

Outputs are written under:

`./runs/<timestamp>/`

- `viewer.html` (open in browser)
- `ocr_report.html` (frame crop + OCR text report)
- `sign_3d_detections.json`
- `crops/*.png` (DETR crop images)
- `annotated_frames/*.png` (frame with bbox/label)
- `vggt_poses.txt`
- `vggt_poses_logs/*.npz` (framewise world pointclouds)

## Python Server

```bash
python ./serve.py --port 8000
```

Then open:

- `http://localhost:8000/index.html`
- `http://localhost:8000/<run_name>/ocr_report.html`

## Reverse Tunnel (optional)

If you need temporary public access without opening firewall/host port:

```bash
python ./open_reverse_tunnel.py --local-port 8000
```

It prints a public `https://...` URL. Keep the process running to keep the tunnel alive.

## Stage-2 Only (if SLAM logs already exist)

```bash
conda run -n sign-map-ros2 python ./pipeline.py \
  --image_folder /path/to/image_sequence_folder \
  --pointcloud_log_dir /path/to/vggt_poses_logs \
  --output_dir /path/to/output_dir \
  --detr_repo ../detr \
  --detr_weights ../detr/outputs/sign_finetune/sign_model_final.pth \
  --ocr_backend auto \
  --ocr_tesseract_lang eng \
  --ocr_numeric_mode \
  --ocr_numeric_only \
  --ocr_numeric_chars "B0123456789 " \
  --ocr_text_pattern '[^B0-9\\s]+'
```

## Notes

- DETR class id for `sign` is set to `1` (matches your fine-tuned setup).
- OCR is `auto`: tries EasyOCR (`ko,en`) first, then falls back to tesseract.
- Tesseract language is requested as `kor+eng`, and auto-resolved based on installed language packs.
- OCR output is cleaned to keep mainly Korean/English/number characters.
- If pointcloud and image resolution differ, bbox coordinates are scaled to pointcloud grid before point extraction.

## ROS2 Bridge

ROS2 package was added at:

- `./ros2_ws/src/sign_map_bridge`

Main nodes:

- `frame_ingest_node`: subscribes RGB(/optional depth), stores frame sequence
- `pipeline_runner_node`: runs this pipeline in CPU mode and publishes new run events
- `result_publisher_node`: publishes trajectory + object markers/details via ROS2 topics

Launch:

```bash
cd ./ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch sign_map_bridge sign_map_stack.launch.py
```

## Video candidates found in this workspace

- `./vggt/examples/videos/kitchen.mp4`
- `./vggt/examples/videos/room.mp4`
- `/workspace/Depth-Anything-3-ori/da3_streaming/IMG_8128.MOV`
