import cv2
import mediapipe as mp

class FaceTracker:
    def __init__(self):
        # Initialize face detection model and drawing utilities
        self.face_detection = mp.solutions.face_detection.FaceDetection()
        self.drawing_utils = mp.solutions.drawing_utils
        self.detected_faces = []

    def process_frame(self, frame):
        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect faces
        result = self.face_detection.process(rgb_frame)
        
        # Clear previously detected faces
        self.detected_faces.clear()
        
        # If faces are detected, draw the detections and store them
        if result.detections:
            for detection in result.detections:
                self.draw_detection(frame, detection)
                self.store_detection(detection, frame.shape)
        
        return frame

    def draw_detection(self, frame, detection):
        # Extract bounding box and confidence score
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
               int(bboxC.width * iw), int(bboxC.height * ih)
        
        # Draw rectangle around the face
        cv2.rectangle(frame, bbox, (255, 0, 0), 2)
        
        # Display the confidence score
        conf_text = f'{int(detection.score[0] * 100)}%'
        cv2.putText(frame, conf_text, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    def store_detection(self, detection, frame_shape):
        # Extract bounding box and confidence score
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame_shape
        bbox = {
            "xmin": int(bboxC.xmin * iw),
            "ymin": int(bboxC.ymin * ih),
            "width": int(bboxC.width * iw),
            "height": int(bboxC.height * ih)
        }
        score = detection.score[0]
        
        # Store the detection data in the list
        self.detected_faces.append({
            "bbox": bbox,
            "score": score
        })

    def get_detected_faces(self):
        # Return the list of detected faces
        return self.detected_faces

    def run(self):
        # Capture video from webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Read a frame from the webcam
            success, frame = cap.read()
            if not success:
                break

            # Process the frame
            frame = self.process_frame(frame)
            
            # Display the frame with face detections
            cv2.imshow("Face Tracker", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

# Create an instance of the FaceTracker class and run it
if __name__ == "__main__":
    face_tracker = FaceTracker()
    face_tracker.run()
    # Get the list of detected faces after running the tracker
    detected_faces = face_tracker.get_detected_faces()
    print(detected_faces)
