import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8 model (YOLOv8n - the nano, smallest and fastest)
model = YOLO('yolov8n.pt')

# Initialize webcam video capture

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame; returns results object
    results = model(frame)

    # Render results on the frame (boxes with labels and confidence)
    annotated_frame = results[0].plot()

    # Show the output frame
    cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

