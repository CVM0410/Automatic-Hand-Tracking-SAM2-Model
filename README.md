# Automatic-Hand-Tracking-SAM2-Model

# SAM 2 + MediaPipe: Hand Tracking and Segmentation

This project detects hands in a video using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html), then provides bounding-box prompts to [SAM 2](https://github.com/facebookresearch/sam2) for segmentation and tracking across all frames.

## Features

- **Automatic Device Selection**: Chooses CUDA if available, then MPS on Apple Silicon, else CPU.
- **MediaPipe Hand Detection**: Uses the first frame to locate hand landmarks (for debugging / bounding‐box generation).
- **SAM 2 Prompt Creation**: Converts the detected hand regions into bounding‐box prompts for SAM 2.
- **Mask Propagation**: SAM 2 propagates the hand masks through every frame of the video.
- **Overlay**: The final segmentation mask is drawn in green over the video and written out as `output_masks.mp4`.
- 

## Installation

1. **Clone This Repo**  
   ```bash
   git clone https://github.com/CVM0410/Automatic-Hand-Tracking-SAM2-Model.git
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   # On Windows use: myenv/Scripts/activate
   ```
   
3. **Install the required Python Libraries**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install SAM2**
   ```bash
   git clone https://github.com/facebookresearch/sam2.git sam2_repo
   cd sam2_repo
   pip install -e .
   cd ..
   ```

5. **Download checkpoint from here**
   Link: https://github.com/facebookresearch/sam2
   Place the file in checkpoints folder.

6. **Run the Video Processing Script**
   ```bash
   python process_video.py
   ```

##Notes

- Ensure your GPU drivers and CUDA are properly installed for faster processing.
- If you face device errors, modify process_video.py to use device="cpu" instead of cuda.
- For large videos, consider reducing batch size in the script.

##References

SAM2: facebookresearch/sam2
MediaPipe: mediapipe.dev
