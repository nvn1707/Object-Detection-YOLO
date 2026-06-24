from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('people.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (640, 384))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video finished!")
        break
    
    results = model(frame)
    annotated = results[0].plot()
    
    out.write(annotated)
    cv2.imshow('Video Object Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Output saved as output_video.mp4!")