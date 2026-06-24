from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

results = model('https://ultralytics.com/images/bus.jpg')

for r in results:
    img = r.plot()
    cv2.imshow('Detection Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()