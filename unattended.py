import cv2
import numpy as np
import time
import threading
import asyncio
import os
from ultralytics import YOLO
from PIL import Image
from telegram import Bot

class UnattendedObjectDetector:
    def __init__(self, model_path="yolov8s.pt"):
        self.model = YOLO(model_path)

        self.BOT_TOKEN = "7324201447:AAHeqKzJDWeG4yOOuv0Jv-NZcylrXYSj39k"
        self.CHAT_ID = "1215055486"
        self.bot = Bot(token=self.BOT_TOKEN)

        self.MIN_CONFIDENCE = 0.10
        self.DIST_THRESHOLD = 150
        self.UNATTENDED_TIMEOUT = 10.0  # Seconds before alert
        self.STATIC_TIMEOUT = 3.0  # Seconds beforeobject static
        self.MISSING_TIMEOUT = 15.0  # Seconds to keep track/ of miss obj
        self.STATIC_MOVE_THRESHOLD = 10  # Pixels
        self.ALERT_COOLDOWN = 10  # Seconds between alerts
        self.FRAME_SKIP = 2  # Process on skip frame 
        
        self.frame_counter = 0
        self.last_processed_frame = None
        
        # STATE
        self.bag_original_owner = {}  # {bag_id: person_id}
        self.bag_owner = {}  #Current owner
        self.bag_timer_start = {}  #When bag became unattended
        self.static_bags = {}  #Bags that haven't moved
        self.last_seen = {}  #Last time object was seen
        self.alerted_bags = set()  #Bags we've already alerted for
        self.last_alert_time = 0  #Time of last alert
        self.unattended_bag_boxes = {}  #For bags that disappeared but were unattended
    
    async def send_telegram_notification(self, image_path, message):
        
        try:
            async with self.bot:
                await self.bot.send_message(chat_id=self.CHAT_ID, text=message)
                with open(image_path, "rb") as photo:
                    await self.bot.send_photo(chat_id=self.CHAT_ID, photo=photo)
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
    
    def send_telegram_alert(self, frame, bag_id):       
        try:
            # Save imag
            image_path = f"alert_snapshot_bag_{bag_id}_{int(time.time())}.jpg"
            cv2.imwrite(image_path, frame)

            message = f"⚠️ Unattended bag detected (Bag ID: {bag_id})"
            threading.Thread(
                target=lambda: asyncio.run(self.send_telegram_notification(image_path, message)),
                daemon=True
            ).start()
            
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
    
    def process_detections(self, frame, current_time):      
        results = self.model.track(source=frame, persist=True, conf=self.MIN_CONFIDENCE,
                                  classes=[0, 24, 26, 28], verbose=False)[0]
        
        persons, bags = {}, {}
        
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls.cpu())
                track_id = int(box.id.cpu()) if box.id is not None else None
                label = self.model.model.names[cls_id]
                
                if label not in ("person", "handbag", "backpack", "suitcase") or track_id is None:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                if label == "person":
                    persons[track_id] = (cx, cy)
                else:
                    bags[track_id] = (cx, cy)
                
                self.last_seen[track_id] = current_time
                
                # Handle static bags
                if label != "person":
                    if track_id not in self.static_bags:
                        self.static_bags[track_id] = {"pos": (cx, cy), "start": current_time}
                    else:
                        old = self.static_bags[track_id]["pos"]
                        dist = ((old[0] - cx)**2 + (old[1] - cy)**2)**0.5
                        if dist < self.STATIC_MOVE_THRESHOLD and current_time - self.static_bags[track_id]["start"] >= self.STATIC_TIMEOUT:
                            bags[track_id] = old  
                        elif dist >= self.STATIC_MOVE_THRESHOLD:
                            self.static_bags[track_id] = {"pos": (cx, cy), "start": current_time}
        
        # Handle disappeared but still within timeout
        for bid in list(self.static_bags):
            if bid not in bags and current_time - self.last_seen.get(bid, 0) < self.MISSING_TIMEOUT:
                bags[bid] = self.static_bags[bid]["pos"]
        
        return persons, bags, results
    
    def update_associations(self, persons, bags, current_time):      
        #associate bags with no owner to nearby persons
        for bag_id, bcent in bags.items():
            if bag_id in self.bag_original_owner:
                continue
                
            best, best_d = None, float("inf")
            for pid, pcent in persons.items():
                d = ((pcent[0] - bcent[0]) ** 2 + (pcent[1] - bcent[1]) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = pid, d
                    
            if best_d < self.DIST_THRESHOLD:
                self.bag_owner[bag_id] = best
                self.bag_original_owner[bag_id] = best
                self.bag_timer_start.pop(bag_id, None)  
        
        #check if bags are still with their owners
        for bag_id, owner_id in list(self.bag_original_owner.items()):
            if bag_id not in bags:
                #Bag disappeared
                if current_time - self.last_seen.get(bag_id, 0) > self.MISSING_TIMEOUT:
                    # Remove from tracking if gone for too long
                    self.bag_original_owner.pop(bag_id, None)
                    self.bag_timer_start.pop(bag_id, None)
                    self.static_bags.pop(bag_id, None)
                continue
            
            if owner_id not in persons:
                #Owner not visible, start/continue timer
                self.bag_timer_start.setdefault(bag_id, current_time)
                continue
            
            # Calculate distance between bag and owner
            d = ((persons[owner_id][0] - bags[bag_id][0]) ** 2 + (persons[owner_id][1] - bags[bag_id][1]) ** 2) ** 0.5
            if d > self.DIST_THRESHOLD:
                #Owner too far start/continue timer
                self.bag_timer_start.setdefault(bag_id, current_time)
            else:
                # Owner close , reset timer
                self.bag_timer_start.pop(bag_id, None)
    
    def draw_annotations(self, frame, results, current_time):
        annotated = results.plot()
        h, w, _ = annotated.shape
        
        # Draw person in green, bag in bluebox
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.model.names[int(box.cls)]
            track_id = int(box.id.cpu()) if box.id is not None else None
            
            col = (0, 255, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, f"ID:{track_id}-{label[0]}", (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            
            # unattended time for left bag
            if label != "person" and track_id in self.bag_timer_start:
                elapsed = current_time - self.bag_timer_start[track_id]
                cv2.putText(annotated, f"Left:{elapsed:.1f}s", (x1, y2 + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        #bag owner associations
        for i, (bag_id, owner_id) in enumerate(self.bag_original_owner.items()):
            text = f"Bag {bag_id} - Owner {owner_id}"
            cv2.putText(annotated, text, (10, 30 + i * 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        #alert
        flash = int(current_time * 2) % 2 == 0 
        for bag_id, start in self.bag_timer_start.items():
            if current_time - start > self.UNATTENDED_TIMEOUT:               
                if flash:
                    cv2.putText(annotated, "!!! UNATTENDED OBJECT ALERT !!!",
                              (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                              (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.rectangle(annotated, (50, 50), (w - 50, h - 50), (0, 0, 255), 4)
                
                #Send alert
                if bag_id not in self.alerted_bags and current_time - self.last_alert_time > self.ALERT_COOLDOWN:
                    self.send_telegram_alert(annotated, bag_id)
                    self.alerted_bags.add(bag_id)
                    self.last_alert_time = current_time
                break
        
        return annotated
    
    def detect_unattended_objects(self, frame):
        current_time = time.time()
        
        self.frame_counter += 1
        
        if self.frame_counter % self.FRAME_SKIP != 0:
            if self.last_processed_frame is not None:
                return self.last_processed_frame


        persons, bags, results = self.process_detections(frame, current_time)

        self.update_associations(persons, bags, current_time)

        annotated_frame = self.draw_annotations(frame, results, current_time)
        self.last_processed_frame = annotated_frame.copy()
        return annotated_frame
