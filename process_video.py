import cv2
import numpy as np
import torch
import mediapipe as mp
import os
from sam2.build_sam import build_sam2_video_predictor

# 1. DEVICE SETUP
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

elif device.type == "mps":
    print("\nWarning: MPS support is preliminary. SAM 2 might run slower or yield different results.")

# 2. MEDIAPIPE SETUP
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hands(image, max_hands=2, conf=0.5):
    # Detects hand landmarks using MediaPipe and returns the processed image and results
    with mp_hands.Hands(static_image_mode=True, max_num_hands=max_hands, min_detection_confidence=conf) as hands:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
    return image, results

# 3. SAM 2 + MEDIAPIPE INTEGRATION
def process_video(input_path, output_path, checkpoint, config):
    # Loads SAM2, detects hands in the first frame using MediaPipe, generates bounding box prompts, applies segmentation, and overlays masks.

    # Load SAM2 model
    predictor = build_sam2_video_predictor(config, checkpoint, device=device.type)
    predictor.to(device)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video:", input_path)
        return
    
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process first frame with MediaPipe
    success, first_frame = cap.read()
    if not success:
        print("Error: No frames found in the video.")
        cap.release()
        return

    processed_frame, results = detect_hands(first_frame)

    # Save intermediate frame for debugging
    cv2.imwrite("hand_landmarks_first_frame.jpg", processed_frame)
    print("Saved hand landmark visualization as 'hand_landmarks_first_frame.jpg'")

    if not results or not results.multi_hand_landmarks:
        print("No hands detected in the first frame.")
        cap.release()
        return

    # Extract bounding boxes
    hand_boxes = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        min_x, max_x = int(min(x_coords) * width), int(max(x_coords) * width)
        min_y, max_y = int(min(y_coords) * height), int(max(y_coords) * height)
        hand_boxes.append((min_x, min_y, max_x, max_y))

    print("Found", len(hand_boxes), "hand(s) in the first frame.")

    with torch.inference_mode():
        # Initialize SAM2 state
        state = predictor.init_state(video_path=input_path, offload_video_to_cpu=False)

        # Create bounding-box prompts
        prompts = []
        for idx, (x1, y1, x2, y2) in enumerate(hand_boxes, start=1):
            box_cpu = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            obj_id_cpu = torch.tensor([idx], dtype=torch.int32)
            prompts.append({"frame_idx": 0, "obj_id": obj_id_cpu, "box": box_cpu})

        # Add bounding-box prompts for first frame
        for p in prompts:
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=p["obj_id"], box=p["box"])

        # Propagate masks and store them
        all_masks = {}
        for f_idx, obj_ids, masks_tensor in predictor.propagate_in_video(state):
            all_masks[f_idx] = masks_tensor.cpu()

    # Overlay masks on video frames
    cap.release()
    cap = cv2.VideoCapture(input_path)

    batch_size = 50
    frame_idx = 0
    batch_frames = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        batch_frames.append(frame_bgr)

        if len(batch_frames) == batch_size or frame_idx == total_frames - 1:
            for i, frame_bgr in enumerate(batch_frames):
                actual_idx = frame_idx - len(batch_frames) + i + 1
                if actual_idx in all_masks:
                    # Combine masks
                    combined_mask = np.zeros((height, width), dtype=np.uint8)
                    masks_arr = all_masks[actual_idx].numpy()  # shape [N, H, W]
                    for single_mask in masks_arr:
                        if single_mask.ndim == 2:
                            combined_mask[single_mask > 0.5] = 255
                        elif single_mask.ndim == 3:
                            combined_mask[single_mask[0] > 0.5] = 255

                    # Apply mask overlay
                    frame_bgr[combined_mask > 0] = (0, 255, 0)

                out_writer.write(frame_bgr)

            batch_frames.clear()
        frame_idx += 1

    cap.release()
    out_writer.release()
    print("Done. Output saved to:", output_path)

# 4. MAIN EXECUTION
if __name__ == "__main__":
    input_video = "test.mp4"
    output_video = "output_masks.mp4"
    checkpoint_path = "sam2.1_hiera_large.pt"
    model_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"

    process_video(input_video, output_video, checkpoint_path, model_cfg_path)
