import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.cap = cv2.VideoCapture(0)
        self.modelhand = mp.solutions.hands
        self.hands = self.modelhand.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mpdraw = mp.solutions.drawing_utils
        self.hand_landmarks_data = []

    def process_frame(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frameRGB)
        if result.multi_hand_landmarks:
            for handLandmarks in result.multi_hand_landmarks:
                self.store_landmarks(handLandmarks, frame)
                for id, lm in enumerate(handLandmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                self.mpdraw.draw_landmarks(frame, handLandmarks, self.modelhand.HAND_CONNECTIONS)
                self.store_landmarks(frame=frame , hand_landmarks=handLandmarks)
        return frame

    def store_landmarks(self, hand_landmarks, frame):
        landmarks = []
        h, w, c = frame.shape
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))
        self.hand_landmarks_data.append(landmarks)



    def start(self):
        while True:
            isSuccess, frame = self.cap.read()
            if not isSuccess:
                break
            
            frame = self.process_frame(frame)
            cv2.imshow("Hand Tracker", frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracker = HandTracker()
    hand_tracker.start()
