import cv2
import csv
from ultralytics import YOLO

# Load YOLOv8-nano
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Create CSV file
csv_file = open("tracking_data.csv", "w", newline="")
writer = csv.writer(csv_file)

# CSV columns
writer.writerow([
    "frame",
    "track_id",
    "class",
    "center_x",
    "center_y",
    "width",
    "height"
])

frame_number = 0

try:
    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam.")
            break

        frame_number += 1

        # YOLO + ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        # Check if objects are being tracked
        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, track_id, class_id in zip(
                boxes, track_ids, class_ids
            ):

                x1, y1, x2, y2 = box

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                width = x2 - x1
                height = y2 - y1

                class_name = model.names[int(class_id)]

                writer.writerow([
                    frame_number,
                    int(track_id),
                    class_name,
                    round(center_x, 2),
                    round(center_y, 2),
                    round(width, 2),
                    round(height, 2)
                ])

        # Make sure data is written immediately
        csv_file.flush()

        # Display tracking
        annotated_frame = result.plot()

        cv2.imshow(
            "YOLOv8 + ByteTrack Tracking",
            annotated_frame
        )

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nTracking stopped using Ctrl+C.")

finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Camera and CSV file closed safely.")