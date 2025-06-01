import cv2
from ultralytics import YOLO
import asyncio
from telegram import Bot
import time
import numpy as np

BOT_TOKEN = "7324201447:AAHeqKzJDWeG4yOOuv0Jv-NZcylrXYSj39k"
CHAT_ID = "1215055486"
bot = Bot(token=BOT_TOKEN)

last_alert_time = 0
alert_cooldown = 30  

def load_model():
    return YOLO("yolov8s.pt")

def detect_overcrowding(frame, model, threshold=10):
    global last_alert_time

    output_frame = frame.copy()

    results = model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    person_count = 0
    person_boxes = []
    
    for box, cls in zip(boxes, classes):
        if int(cls) == 0: 
            person_count += 1
            person_boxes.append(box)
    
    alert = person_count > threshold
    
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) 
        
        if alert:
            color = (0, 0, 255) 
            
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
    #disply count
    cv2.putText(output_frame, f"People Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display alert
    if alert:
        cv2.putText(output_frame, "ALERT: OVERCROWDING DETECTED!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        current_time = time.time()
        if current_time - last_alert_time > alert_cooldown:
            send_telegram_alert(output_frame, person_count)
            last_alert_time = current_time
    
    return output_frame

def send_telegram_alert(frame, person_count):
    image_path = "overcrowding_alert.jpg"
    cv2.imwrite(image_path, frame)
    
    try:
        message = f"🚨 OVERCROWDING ALERT! {person_count} people detected"
        asyncio.run(send_telegram_notification(image_path, message))
    except Exception as e:
        print(f"Telegram error: {e}")

async def send_telegram_notification(image_path, message):
    async with bot:
        await bot.send_message(chat_id=CHAT_ID, text=message)
        with open(image_path, "rb") as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo)

def main():

    model = load_model()

    cap = cv2.VideoCapture(0)  

    overcrowding_threshold = 3 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detect_overcrowding(frame, model, overcrowding_threshold)
        
        cv2.imshow("Overcrowding Detection", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
