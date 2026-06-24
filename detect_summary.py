from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

print("=" * 50)
print("OBJECT DETECTION WITH YOLO - PROJECT SUMMARY")
print("=" * 50)

print("\n Testing on image...")
results = model('bus.jpg')
for r in results:
    boxes = r.boxes
    print(f" Objects detected in image: {len(boxes)}")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        print(f"   - {name}: {conf:.0%} confidence")

print("\n Done! Check output for results.")
print("=" * 50)