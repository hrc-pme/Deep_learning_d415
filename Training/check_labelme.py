import json
import cv2
import numpy as np

# Path to the LabelMe JSON file and the image
json_file = "../cable_dataset/train_added/2024_1202/ac_cable_83.json"  # Update with the path to your LabelMe JSON
image_file = "../cable_dataset/train_added/2024_1202/ac_cable_83.jpg"  # Update with the path to the corresponding image

# Load the LabelMe JSON annotations
with open(json_file) as f:
    labelme_data = json.load(f)

# Load the image
image = cv2.imread(image_file)

# Loop through the shapes (annotations) in the LabelMe data
for shape in labelme_data['shapes']:
    label = shape['label']
    points = np.array(shape['points'], dtype=np.int32)
    
    # Check if the annotation is a polygon or a bounding box
    if len(points) == 2:  # Bounding box (2 points)
        # Assuming points[0] is the top-left and points[1] is the bottom-right
        top_left = tuple(points[0])
        bottom_right = tuple(points[1])
        print(points)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green bounding box
    else:  # Polygon
        points = points.reshape((-1, 1, 2))  # Convert to correct shape for poly drawing
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue polygon

    # Optionally, put the label text near the annotation
    cv2.putText(image, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Show the image with annotations
cv2.imshow("LabelMe Annotations", image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
