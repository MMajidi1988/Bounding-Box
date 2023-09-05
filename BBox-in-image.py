from ultralytics import YOLO
import cv2

model = YOLO("/home/mehdi/AIproducer/YOLO-Object-Detection/YOLO-Model/yolov8m-football.pt")

# Define class names (replace with actual names if available)
class_names = ['ball', 'class_1', 'player', 'class_3', 'class_4']

image_path = "/home/mehdi/AIproducer/AI-Producer/aiproducer-new_feature_smart_crop/classification_service/frames/frame_5.jpg"
results = model(image_path)
boxes = results[0].cpu().numpy()

img = cv2.imread(image_path)

for box in boxes:
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    conf = box[4]  # confidence score
    cls = box[5]   # class index

    # Print out the raw results
    print(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2}), Confidence score: {conf}, Class: {class_names[int(cls)]}")

    if conf > 0.5:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f'{class_names[int(cls)]}: {conf:.2f}'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y1 = max(y1, labelSize[1])
        cv2.rectangle(img, (x1, y1 - labelSize[1]), (x1 + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.imwrite("/home/mehdi/AIproducer/YOLO-Object-Detection/Images-output/4.jpeg", img)
