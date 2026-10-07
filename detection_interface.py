from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def detect_objects(frame):
    results = model(frame, verbose=False)

    detections = []

    result = results[0]

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for bbox, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids
        ):

            x1, y1, x2, y2 = bbox

            detections.append({
                "class_name": model.names[int(class_id)],
                "confidence": float(confidence),
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            })

    return detections