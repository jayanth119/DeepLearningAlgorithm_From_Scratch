import cv2
import mediapipe as mp

class FaceMeshTracker:
    def __init__(self):
        # Initialize face mesh model
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                         max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=0.5,
                                                         min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        self.detected_faces = []

    def process_frame(self, frame):
        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect face mesh
        result = self.face_mesh.process(rgb_frame)
        
        # Clear previously detected faces
        self.detected_faces.clear()
        
        # If face mesh is detected, draw the mesh and store landmarks
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                self.draw_mesh(frame, face_landmarks)
                self.store_landmarks(face_landmarks)
        
        return frame

    def draw_mesh(self, frame, face_landmarks):
        # Draw face mesh landmarks
        self.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.drawing_spec)

    def store_landmarks(self, face_landmarks):
        # Store the face landmarks in the list
        landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_landmarks.landmark]
        self.detected_faces.append(landmarks)

    def get_detected_faces(self):
        # Return the list of detected faces landmarks
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
            
            # Display the frame with face mesh
            cv2.imshow("Face Mesh Tracker", frame)#468 marks 
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

# Create an instance of the FaceMeshTracker class and run it
if __name__ == "__main__":
    face_mesh_tracker = FaceMeshTracker()
    face_mesh_tracker.run()
    # Get the list of detected faces landmarks after running the tracker
    detected_faces = face_mesh_tracker.get_detected_faces()
    print(detected_faces)
