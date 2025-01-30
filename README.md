# Automatic-Hand-Tracking-SAM2-Model

# SAM 2 + MediaPipe: Hand Tracking and Segmentation

This project detects hands in a video using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html), then provides bounding-box prompts to [SAM 2](https://github.com/facebookresearch/sam2) for segmentation and tracking across all frames.

## Features

- **Automatic Device Selection**: Chooses CUDA if available, then MPS on Apple Silicon, else CPU.
- **MediaPipe Hand Detection**: Uses the first frame from the video `test.mp4` to locate hand landmarks and save a debugging image `hand_landmarks_first_frame.jpg` to visualize detected hand positions.
- **SAM 2 Prompt Creation**: Converts the detected hand regions into bounding‐box prompts for SAM 2.
- **Mask Propagation**: SAM 2 propagates the hand masks through every frame of the video.
- **Overlay**: The final segmentation mask is drawn in green over the video and written out as `output_masks.mp4`.

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

5. **Download model checkpoint from here**
   - Link: [SAM2](https://github.com/facebookresearch/sam2)
   - Place the downloaded checkpoint file in ```checkpoints/``` folder

6. **Run the Video Processing Script**
   ```bash
   python process_video.py
   ```

## Common Issues & Fixes

🔴 Issue: FileNotFoundError for test.mp4
✅ Fix: Ensure the file is in the same directory as the script or provide the absolute path.

🔴 Issue: checkpoints/sam2.1_hiera_large.pt Not Found
✅ Fix: Download and place the model file in the checkpoints/ directory.

🔴 Issue: CUDA Out of Memory Error
✅ Fix: Lower the batch size inside the script

## Notes

- Ensure your GPU drivers and CUDA are properly installed for faster processing.
- If you face device errors, modify process_video.py to use device="cpu" instead of cuda.
- For large videos, consider reducing batch size in the script.

## References

- [SAM2](https://github.com/facebookresearch/sam2)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
