from ultralytics import YOLO
import cv2
import requests
import time

# ------------------ Telegram Setup ------------------
BOT_TOKEN = "YOUR-BOT-TOKEN"
CHAT_ID = "YOUR-CHAT-ID"

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": text
    }
    requests.post(url, data=data)


def send_telegram_photo(photo_path, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    with open(photo_path, "rb") as photo:
        files = {"photo": photo}
        data = {
            "chat_id": CHAT_ID,
            "caption": caption
        }
        requests.post(url, data=data, files=files)


# ------------------ YOLO Setup ------------------
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

last_alert_time = 0
cooldown = 120   # seconds (2 minutes)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    human_detected = False

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if model.names[cls_id] == "person" and confidence > 0.6:

                human_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                cv2.putText(
                    frame,
                    f"Human {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # ------------------ Telegram Alert Trigger ------------------

    current_time = time.time()

    if human_detected and (current_time - last_alert_time > cooldown):

        print("Human detected! Sending Telegram alert...")

        image_path = "snapshot.jpg"
        cv2.imwrite(image_path, frame)

        send_telegram_message("⚠ ALERT: Human detected by AI system!")
        send_telegram_photo(image_path, "Detection Snapshot")

        last_alert_time = current_time


    cv2.imshow("YOLOv8 Human Detection - Code to Reality", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
