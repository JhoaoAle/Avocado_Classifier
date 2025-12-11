from ultralytics import YOLO
import cv2

# Load trained segmentation model
model = YOLO("runs/segment/train8/weights/best.pt")

#image_path = "Aguacate-hass-un_123077.jpg"
#image_path = "5c8593bb4c6d1.jpeg"
image_path = "Aguacate-hass-un_123077.jpg"
results = model(image_path)[0]


# Plot segmentation overlays
annotated = results.plot()

cv2.imshow("Segmentation Results", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
