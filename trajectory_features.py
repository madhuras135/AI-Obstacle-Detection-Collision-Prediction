import csv
import math

# Input and output files
input_file = "tracking_data.csv"
output_file = "trajectory_features.csv"

# Store previous position of each tracked object
previous_positions = {}

with open(input_file, "r", newline="") as infile, \
     open(output_file, "w", newline="") as outfile:

    reader = csv.DictReader(infile)

    fieldnames = [
        "frame",
        "track_id",
        "class",
        "center_x",
        "center_y",
        "width",
        "height",
        "dx",
        "dy",
        "speed",
        "direction"
    ]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:

        frame = int(row["frame"])
        track_id = int(row["track_id"])

        center_x = float(row["center_x"])
        center_y = float(row["center_y"])

        # Default movement for a newly appearing object
        dx = 0.0
        dy = 0.0

        # Calculate movement if this object appeared in the previous frame
        if track_id in previous_positions:

            previous_x, previous_y = previous_positions[track_id]

            dx = center_x - previous_x
            dy = center_y - previous_y

        # Calculate movement magnitude
        speed = math.sqrt(dx ** 2 + dy ** 2)

        # Determine basic movement direction
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

        # Save current position for next frame
        previous_positions[track_id] = (center_x, center_y)

        writer.writerow({
            "frame": frame,
            "track_id": track_id,
            "class": row["class"],
            "center_x": round(center_x, 2),
            "center_y": round(center_y, 2),
            "width": row["width"],
            "height": row["height"],
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "speed": round(speed, 2),
            "direction": direction
        })

print("Trajectory feature extraction completed.")
print("Created:", output_file)