from ultralytics import YOLO
import math
import cv2
import cvzone
import torch
from image_to_text import predict_number_plate
from paddleocr import PaddleOCR
import os

image_path = "CAPTURE.JPG" 
output_folder = "outputs"  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = YOLO(r"C:\SAFERIDE\runs\detect\train2\weights\best.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classNames = ["with helmet", "without helmet", "rider", "number plate"]
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def group_nearby_riders(rider_boxes, max_distance=150):
    groups = []
    processed = set()
    
    for i, box1 in enumerate(rider_boxes):
        if i in processed:
            continue
            
        x1, y1, x2, y2 = box1
        center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
        current_group = [box1]
        processed.add(i)
        
        for j, box2 in enumerate(rider_boxes):
            if j in processed or j == i:
                continue
                
            x1_2, y1_2, x2_2, y2_2 = box2
            center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
        
            distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
            
            if distance < max_distance:
                current_group.append(box2)
                processed.add(j)
        
        groups.append(current_group)
    
    return groups

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"could not read image {image_path}")
        return
    
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=device)
    
    for r in results:
        boxes = r.boxes
        li = dict()
        rider_box = list()
        
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        
        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
            indices = torch.where(new_boxes[:, -1] == 2)
            rows = new_boxes[indices]
            for box in rows:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rider_box.append((x1, y1, x2, y2))
        except:
            pass
        
        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box[4] * 100)) / 100
            cls = int(box[5])
            
            if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or \
                    classNames[cls] == "number plate" and conf >= 0.5:
                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))
                
                if rider_box:
                    for j, rider in enumerate(rider_box):
                        if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
                                y2 <= rider_box[j][3]:
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                              offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                            
                            li.setdefault(f"rider{j}", [])
                            li[f"rider{j}"].append(classNames[cls])
                            
                            if classNames[cls] == "number plate":
                                npx, npy, npw, nph, npconf = x1, y1, w, h, conf
                                crop = img[npy:npy + h, npx:npx + w]
                                
                        if li:
                            for key, value in li.items():
                                if key == f"rider{j}":
                                    if len(list(set(li[f"rider{j}"]))) == 3:
                                        try:
                                            vechicle_number, conf = predict_number_plate(crop, ocr)
                                            if vechicle_number and conf:
                                                cvzone.putTextRect(img, f"{vechicle_number} {round(conf*100, 2)}%",
                                                                  (x1, y1 - 50), scale=1.5, offset=10,
                                                                  thickness=2, colorT=(39, 40, 41),
                                                                  colorR=(105, 255, 255))
                                        except Exception as e:
                                            print(e)

        if rider_box:
            rider_groups = group_nearby_riders(rider_box)
            
            for group in rider_groups:
                if len(group) >= 3:
                    min_x = min(box[0] for box in group)
                    min_y = min(box[1] for box in group)
                    max_x = max(box[2] for box in group)
                    max_y = max(box[3] for box in group)

                    group_w, group_h = max_x - min_x, max_y - min_y
                    cvzone.cornerRect(img, (min_x, min_y, group_w, group_h), l=15, rt=5, colorR=(0, 0, 255))  # Red for triples
                    cvzone.putTextRect(img, f"TRIPLE RIDERS ({len(group)})", (min_x + 10, min_y - 10), scale=1.5,
                                      offset=10, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))  # Red background with white text
    

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")

    cv2.imshow('Processed Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing {image_path}...")
            process_image(image_path)




if os.path.isdir(image_path):
    process_directory(image_path)
else:
    process_image(image_path)

#RISHI