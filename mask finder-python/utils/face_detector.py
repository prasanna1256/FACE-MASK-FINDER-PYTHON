import cv2
import numpy as np
import os

class FaceDetector:
    """
    Class to handle face detection using OpenCV's Haar Cascade classifier
    """
    def __init__(self, cascade_path=None):
        """
        Initialize the face detector
        Args:
            cascade_path: Path to the Haar cascade XML file. If None, uses cv2's default
        """
        if cascade_path is None:
            # Try to use OpenCV's built-in Haar cascade for face detection
            haar_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(haar_model_path):
                cascade_path = haar_model_path
            else:
                raise FileNotFoundError(f"Default Haar cascade file not found at {haar_model_path}")

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        """
        Detect faces in an image
        Args:
            image: Input image (BGR format as used by OpenCV)
        Returns:
            List of (x, y, w, h) tuples for face bounding boxes
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return faces

    def extract_face_regions(self, image, faces, padding=10):
        """
        Extract face regions from image based on face detection results
        Args:
            image: Input image
            faces: List of (x, y, w, h) tuples for face bounding boxes
            padding: Extra padding to add around detected faces
        Returns:
            List of (face_img, (x, y, w, h)) tuples
        """
        face_regions = []
        h, w = image.shape[:2]

        for (x, y, fw, fh) in faces:
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + fw + padding)
            y2 = min(h, y + fh + padding)

            # Extract the face region
            face_img = image[y1:y2, x1:x2]
            face_regions.append((face_img, (x1, y1, x2-x1, y2-y1)))

        return face_regions

    def draw_faces(self, image, faces, results=None, colors=None):
        """
        Draw bounding boxes around detected faces, optionally with prediction results
        Args:
            image: Input image
            faces: List of (x, y, w, h) tuples for face bounding boxes
            results: Optional list of prediction results (same length as faces)
            colors: Optional dictionary mapping result labels to BGR colors
        Returns:
            Image with bounding boxes drawn
        """
        img_copy = image.copy()

        if colors is None:
            colors = {
                'with_mask': (0, 255, 0),  # Green
                'without_mask': (0, 0, 255)  # Red
            }

        # Default color for boxes (if no results provided)
        default_color = (255, 255, 0)  # Yellow

        for i, (x, y, w, h) in enumerate(faces):
            color = default_color
            label = ""

            # If we have prediction results
            if results is not None and i < len(results):
                result, confidence = results[i]
                color = colors.get(result, default_color)
                label = f"{result}: {confidence*100:.2f}%"

            # Draw rectangle around face
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)

            # If we have a label, add it
            if label:
                y_offset = max(y - 10, 0)
                cv2.putText(img_copy, label, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        return img_copy
