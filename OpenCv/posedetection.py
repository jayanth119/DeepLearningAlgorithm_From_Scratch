import cv2
import mediapipe as mp

class PoseTracker:
    def __init__(self, video_path, detection_confidence=0.5, tracking_confidence=0.5, screen_size=(640, 480)):
        self.cap = cv2.VideoCapture(video_path)
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mpdraw = mp.solutions.drawing_utils
        self.pose_landmarks_data = []
        self.screen_size = screen_size

    def process_frame(self, frame):
        imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(imgrgb)
        if result.pose_landmarks:
            self.store_landmarks(result.pose_landmarks, frame)
            self.mpdraw.draw_landmarks(frame, result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return frame

    def store_landmarks(self, pose_landmarks, frame):
        landmarks = []
        h, w, c = frame.shape
        for lm in pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))
        self.pose_landmarks_data.append(landmarks)

    def start(self):
        while True:
            isSuccess, frame = self.cap.read()
            if not isSuccess:
                break
            
            frame = self.process_frame(frame)
            frame = cv2.resize(frame, self.screen_size)
            cv2.imshow("Pose Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "One dance - Velocity edit.mp4"
    pose_tracker = PoseTracker(video_path, screen_size=(800, 600))  # Specify the desired screen size
    pose_tracker.start()
