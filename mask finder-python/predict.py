import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.face_detector import FaceDetector
from utils.data_preprocessing import preprocess_image

class MaskDetector:
    """
    Class for detecting face masks in images and videos
    """
    def __init__(self, model_path, classes_path=None, img_size=(150, 150)):
        """
        Initialize the mask detector
        Args:
            model_path: Path to the trained model file
            classes_path: Path to the saved class labels file
            img_size: Image size used for the model
        """
        # Load model
        self.model = load_model(model_path)
        self.img_size = img_size

        # Load class labels
        if classes_path is None:
            model_dir = os.path.dirname(model_path)
            classes_path = os.path.join(model_dir, "classes.npy")

        if os.path.exists(classes_path):
            self.classes = np.load(classes_path, allow_pickle=True)
        else:
            # Default class labels
            self.classes = np.array(['with_mask', 'without_mask'])

        # Initialize face detector
        self.face_detector = FaceDetector()

    def predict_mask(self, face_img):
        """
        Predict if a face is wearing a mask
        Args:
            face_img: Image of a face
        Returns:
            (class_label, confidence) tuple
        """
        # Resize image to match model input size
        img = cv2.resize(face_img, self.img_size)

        # Normalize pixel values
        img = img.astype("float32") / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Make prediction
        preds = self.model.predict(img)[0]

        # Get the predicted class and confidence
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        label = self.classes[class_idx]

        return label, confidence

    def process_image(self, image):
        """
        Process an image to detect faces and predict mask usage
        Args:
            image: Input image (BGR format)
        Returns:
            Annotated image, list of detection results
        """
        # Convert BGR to RGB (for model input)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.face_detector.detect_faces(image)

        # Extract face regions
        face_regions = self.face_detector.extract_face_regions(rgb_image, faces)

        # Process each face
        results = []
        for face_img, _ in face_regions:
            # Predict mask usage
            label, confidence = self.predict_mask(face_img)
            results.append((label, confidence))

        # Draw results on the image
        annotated_image = self.face_detector.draw_faces(image, faces, results)

        return annotated_image, results, faces

    def process_image_file(self, image_path):
        """
        Process an image file to detect faces and predict mask usage
        Args:
            image_path: Path to the image file
        Returns:
            Annotated image, list of detection results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Process image
        return self.process_image(image)

def main(model_path, image_path):
    """
    Main function to test mask detection on a single image
    """
    # Initialize mask detector
    detector = MaskDetector(model_path)

    # Process image
    annotated_img, results, faces = detector.process_image_file(image_path)

    # Print results
    print(f"Detected {len(faces)} faces:")
    for i, (label, confidence) in enumerate(results):
        print(f"  Face {i+1}: {label} (confidence: {confidence*100:.2f}%)")

    # Display result
    cv2.imshow("Result", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    output_path = "result_" + os.path.basename(image_path)
    cv2.imwrite(output_path, annotated_img)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect face masks in an image")
    parser.add_argument("--model", default="./models/mask_detector_model.h5",
                        help="Path to trained model file")
    parser.add_argument("--image", required=True,
                        help="Path to input image file")

    args = parser.parse_args()
    main(args.model, args.image)
