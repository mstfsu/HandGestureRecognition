import cv2
import numpy as np
from tensorflow.keras import models
import time
import mediapipe as mp

# Load the trained model
model = models.load_model('hand_gesture_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def get_hand_from_frame(image):
    converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(converted_img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the bounding box of the hand.
            x_max = 0
            y_max = 0
            x_min = float('inf')
            y_min = float('inf')

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y

            padding = 10
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, image.shape[1])
            y_max = min(y_max + padding, image.shape[0])

            hand_image = image[y_min:y_max, x_min:x_max]
            return hand_image


def preprocess_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (64, 64))
    img_normalized = resized_img.astype(np.float32) / 255.0
    img_with_channel = np.expand_dims(img_normalized, axis=-1)
    return np.expand_dims(img_with_channel, axis=0)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        hand_img = get_hand_from_frame(frame)

        if hand_img is not None:
            preprocessed_img = preprocess_image(hand_img)
            assert preprocessed_img.shape == (
                1, 64, 64, 1), f"Unexpected shape of preprocessed image: {preprocessed_img.shape}"
            prediction = model.predict(preprocessed_img)
            class_index = np.argmax(prediction)
            label = {
                0: "palm",
                1: "l",
                2: "fist",
                3: "fist_moved",
                4: "thumb",
                5: "index",
                6: "ok",
                7: "palm_moved",
                8: "c",
                9: "down"
            }.get(class_index, 'Unknown')
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)
        continue

cap.release()
cv2.destroyAllWindows()


# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
