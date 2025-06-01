import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from telegram import Bot
import time
import asyncio

BOT_TOKEN = "7324201447:AAHeqKzJDWeG4yOOuv0Jv-NZcylrXYSj39k"
CHAT_ID = "1215055486"
bot = Bot(token=BOT_TOKEN)

last_alert_time = 0
alert_cooldown = 10  

def load_model():
    return YOLO("yolov8n.pt")

def is_point_inside_polygon(point, polygon):
    return Polygon(polygon).contains(Point(point))

def detect_intrusion(frame, model, roi_points):

    global last_alert_time
    
    output_frame = frame.copy()
    
    # Draw restricted
    if len(roi_points) > 2:
        overlay = output_frame.copy()
        cv2.polylines(overlay, [np.array(roi_points)], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.fillPoly(overlay, [np.array(roi_points)], color=(0, 165, 255))
        alpha = 0.3 
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
    
    results = model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    
    alert = False
    
    #detecting
    for box, cls in zip(boxes, classes):
        if int(cls) != 0: 
            continue
            
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = (0, 255, 0) 
        
        #person is inside restricted area
        if is_point_inside_polygon((cx, cy), roi_points):
            color = (0, 0, 255) 
            alert = True
            cv2.putText(output_frame, "INTRUSION!", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output_frame, (cx, cy), 4, color, -1)
    

    if alert:
        cv2.putText(output_frame, "ALERT: INTRUSION DETECTED!", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        # Send Telegram alert if cooldown has passed
        current_time = time.time()
        if current_time - last_alert_time > alert_cooldown:
            send_telegram_alert(output_frame)
            last_alert_time = current_time
    
    return output_frame

def send_telegram_alert(frame):
    image_path = "intrusion_alert.jpg"
    cv2.imwrite(image_path, frame)
    
    try:
        asyncio.run(send_telegram_notification(image_path))
    except Exception as e:
        print(f"Telegram error: {e}")

async def send_telegram_notification(image_path):
    async with bot:
        await bot.send_message(chat_id=CHAT_ID, text="🚨 Intrusion Detected!")
        with open(image_path, "rb") as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo)

def main():    
    model = load_model()
    
    cap = cv2.VideoCapture(0)  

    roi_points = []
    polygon_complete = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, polygon_complete
        if event == cv2.EVENT_LBUTTONDOWN and not polygon_complete:
            roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(roi_points) > 2:
            polygon_complete = True

    cv2.namedWindow("Intrusion Detection")
    cv2.setMouseCallback("Intrusion Detection", mouse_callback)
    
    print("[INFO] Left-click to mark ROI points, right-click to finish drawing.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if not polygon_complete:
            for pt in roi_points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            
            if len(roi_points) > 1:
                cv2.polylines(frame, [np.array(roi_points)], isClosed=False, color=(0, 255, 255), thickness=2)
            
            cv2.putText(frame, "Draw ROI - Left click to add points, Right click to finish", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Intrusion Detection", frame)
        else:
            processed_frame = detect_intrusion(frame, model, roi_points)
            cv2.imshow("Intrusion Detection", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == 27: 
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
