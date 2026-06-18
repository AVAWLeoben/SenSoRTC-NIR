import cv2
from tqdm import tqdm  # pip install tqdm
from pathlib import Path

# === CONFIG ===
input_video = "simulated_conveyor.mp4"
overlay_image = Path("pizza.jpg")
output_video = f"output_{overlay_image.stem}.mp4"

black_thresh = 40
edge_pad = 3
scale_factor = 2.0
output_size = 640              # final video resolution
min_overlay_size = 128          # minimum injected image size (pixels)

# === LOAD OVERLAY IMAGE ===
overlay = cv2.imread(str(overlay_image), cv2.IMREAD_UNCHANGED)

if overlay is None:
    raise FileNotFoundError(f"Could not load overlay image: {overlay_image}")

oh, ow = overlay.shape[:2]
aspect = ow / oh

# === VIDEO SETUP ===
cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (output_size, output_size)
)

# === PROCESS VIDEO ===
with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize source frame first to save memory + match YOLO size
        frame = cv2.resize(
            frame,
            (output_size, output_size),
            interpolation=cv2.INTER_AREA
        )

        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect black boxes
        mask = cv2.threshold(
            gray,
            black_thresh,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Ignore tiny noise
            if w < 20 or h < 20:
                continue

            # Expand slightly
            x2 = max(0, x - edge_pad)
            y2 = max(0, y - edge_pad)
            x3 = min(width, x + w + edge_pad)
            y3 = min(height, y + h + edge_pad)

            box_w = x3 - x2
            box_h = y3 - y2

            # Remove black box
            frame[y2:y3, x2:x3] = (255, 255, 255)

            # Preserve aspect ratio
            new_h = int(box_h * scale_factor)
            new_w = int(new_h * aspect)

            max_w = int(box_w * scale_factor)

            if new_w > max_w:
                new_w = max_w
                new_h = int(new_w / aspect)

            # Enforce minimum size
            if new_w < min_overlay_size:
                new_w = min_overlay_size
                new_h = int(new_w / aspect)

            if new_h < min_overlay_size:
                new_h = min_overlay_size
                new_w = int(new_h * aspect)

            center_x = x2 + box_w // 2
            center_y = y2 + box_h // 2

            start_x = center_x - new_w // 2
            start_y = center_y - new_h // 2
            end_x = start_x + new_w
            end_y = start_y + new_h

            # Clip to frame
            crop_x1 = max(0, start_x)
            crop_y1 = max(0, start_y)
            crop_x2 = min(width, end_x)
            crop_y2 = min(height, end_y)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            resized = cv2.resize(
                overlay,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )

            # Crop overlay if needed
            overlay_x1 = crop_x1 - start_x
            overlay_y1 = crop_y1 - start_y
            overlay_x2 = overlay_x1 + (crop_x2 - crop_x1)
            overlay_y2 = overlay_y1 + (crop_y2 - crop_y1)

            resized_crop = resized[
                overlay_y1:overlay_y2,
                overlay_x1:overlay_x2
            ]

            # Apply overlay
            if resized_crop.shape[2] == 4:
                alpha = resized_crop[:, :, 3] / 255.0

                for c in range(3):
                    frame[crop_y1:crop_y2, crop_x1:crop_x2, c] = (
                        alpha * resized_crop[:, :, c]
                        + (1 - alpha)
                        * frame[crop_y1:crop_y2, crop_x1:crop_x2, c]
                    )
            else:
                frame[crop_y1:crop_y2, crop_x1:crop_x2] = (
                    resized_crop[:, :, :3]
                )

        out.write(frame)
        pbar.update(1)

# === CLEANUP ===
cap.release()
out.release()

print(f"Done! Saved as {output_video}")