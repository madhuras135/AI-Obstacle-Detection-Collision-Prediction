import cv2
import csv
import math
import torch
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. LOAD MODELS
# ============================================================

print("Loading YOLOv8-nano...")

yolo = YOLO("yolov8n.pt")

print("YOLOv8-nano loaded.")


print("Loading MiDaS-small...")

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)

midas.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform

print("MiDaS-small loaded.")


# ============================================================
# 2. SELECT INPUT SOURCE
# ============================================================

print("\nChoose input source:")
print("1. Webcam")
print("2. Video file")

choice = input("Enter choice (1 or 2): ").strip()


if choice == "1":

    print("Opening webcam...")

    cap = cv2.VideoCapture(0)

    input_name = "webcam"


elif choice == "2":

    video_name = input(
        "Enter video filename from the videos folder: "
    ).strip()

    video_path = "videos/" + video_name

    print("Opening video:", video_path)

    cap = cv2.VideoCapture(video_path)

    input_name = video_name


else:

    print("Invalid choice.")
    exit()


if not cap.isOpened():

    print("Could not open the selected input.")
    exit()


# ============================================================
# 3. CREATE CSV
# ============================================================

csv_file = open(
    "integrated_tracking_data.csv",
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "frame",
    "track_id",
    "class",
    "center_x",
    "center_y",
    "width",
    "height",
    "relative_depth",
    "depth_change",
    "dx",
    "dy",
    "speed",
    "direction"
])


# ============================================================
# 4. STORE PREVIOUS VALUES
# ============================================================

previous_positions = {}
previous_depths = {}

frame_number = 0


# ============================================================
# 5. MAIN LOOP
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success:

            print("\nInput finished or could not be read.")
            break

        frame_number += 1


        # ====================================================
        # A. MiDaS RELATIVE DEPTH
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        input_batch = transform(rgb)

        with torch.no_grad():

            prediction = midas(input_batch)


        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()


        depth_map = prediction.cpu().numpy()


        # ====================================================
        # B. YOLO + BYTETRACK
        # ====================================================

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]


        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()

            track_ids = result.boxes.id.cpu().numpy()

            class_ids = result.boxes.cls.cpu().numpy()


            for box, track_id, class_id in zip(
                boxes,
                track_ids,
                class_ids
            ):

                x1, y1, x2, y2 = box

                track_id = int(track_id)

                class_id = int(class_id)


                # ============================================
                # OBJECT POSITION
                # ============================================

                center_x = (x1 + x2) / 2

                center_y = (y1 + y2) / 2

                width = x2 - x1

                height = y2 - y1

                class_name = yolo.names[class_id]


                # ============================================
                # MOVEMENT
                # ============================================

                dx = 0.0

                dy = 0.0


                if track_id in previous_positions:

                    old_x, old_y = previous_positions[track_id]

                    dx = center_x - old_x

                    dy = center_y - old_y


                speed = math.sqrt(
                    dx ** 2 + dy ** 2
                )


                # ============================================
                # MOVEMENT DIRECTION
                # ============================================

                if abs(dx) < 1 and abs(dy) < 1:

                    direction = "stationary"


                elif abs(dx) > abs(dy):

                    if dx > 0:

                        direction = "right"

                    else:

                        direction = "left"


                else:

                    if dy > 0:

                        direction = "down"

                    else:

                        direction = "up"


                # ============================================
                # OBJECT RELATIVE DEPTH
                # ============================================

                # Keep coordinates inside image

                x1_int = max(
                    0,
                    int(x1)
                )

                y1_int = max(
                    0,
                    int(y1)
                )

                x2_int = min(
                    frame.shape[1],
                    int(x2)
                )

                y2_int = min(
                    frame.shape[0],
                    int(y2)
                )


                if (
                    x2_int > x1_int
                    and y2_int > y1_int
                ):

                    # Use the central part of the
                    # bounding box to reduce
                    # background influence

                    box_width = x2_int - x1_int

                    box_height = y2_int - y1_int


                    cx1 = (
                        x1_int
                        + int(box_width * 0.25)
                    )

                    cx2 = (
                        x1_int
                        + int(box_width * 0.75)
                    )


                    cy1 = (
                        y1_int
                        + int(box_height * 0.25)
                    )

                    cy2 = (
                        y1_int
                        + int(box_height * 0.75)
                    )


                    depth_region = depth_map[
                        cy1:cy2,
                        cx1:cx2
                    ]


                    if depth_region.size > 0:

                        relative_depth = float(
                            np.median(depth_region)
                        )

                    else:

                        relative_depth = 0.0


                else:

                    relative_depth = 0.0


                # ============================================
                # DEPTH CHANGE
                # ============================================

                depth_change = 0.0


                if track_id in previous_depths:

                    old_depth = previous_depths[track_id]

                    depth_change = (
                        relative_depth
                        - old_depth
                    )


                # ============================================
                # SAVE CURRENT VALUES
                # ============================================

                previous_positions[track_id] = (
                    center_x,
                    center_y
                )


                previous_depths[track_id] = (
                    relative_depth
                )


                # ============================================
                # WRITE CSV
                # ============================================

                writer.writerow([
                    frame_number,
                    track_id,
                    class_name,
                    round(center_x, 2),
                    round(center_y, 2),
                    round(width, 2),
                    round(height, 2),
                    round(relative_depth, 4),
                    round(depth_change, 4),
                    round(dx, 2),
                    round(dy, 2),
                    round(speed, 2),
                    direction
                ])


        # ====================================================
        # SAVE DATA CONTINUOUSLY
        # ====================================================

        csv_file.flush()


        # ====================================================
        # DISPLAY
        # ====================================================

        annotated_frame = result.plot()


        cv2.imshow(
            "YOLO + ByteTrack + MiDaS",
            annotated_frame
        )


        # Press Q to stop

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("\nStopped by user.")

            break


except KeyboardInterrupt:

    print("\nTracking stopped using Ctrl+C.")


finally:

    cap.release()

    csv_file.close()

    cv2.destroyAllWindows()

    print("\nInput closed.")

    print(
        "CSV saved as integrated_tracking_data.csv"
    )