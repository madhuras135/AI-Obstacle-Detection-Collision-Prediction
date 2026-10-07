import torch
import cv2

print("Loading MiDaS-small...")

# Load MiDaS-small
model_type = "MiDaS_small"

midas = torch.hub.load(
    "intel-isl/MiDaS",
    model_type
)

midas.eval()

print("MiDaS-small loaded successfully!")

# Load MiDaS transforms
midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    # Convert BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare image
    input_batch = transform(img)

    # Predict depth
    with torch.no_grad():
        prediction = midas(input_batch)

    # Resize depth map to camera resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize for visualization
    depth_normalized = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_normalized = depth_normalized.astype("uint8")

    # Display depth
    cv2.imshow("MiDaS-small Relative Depth", depth_normalized)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()