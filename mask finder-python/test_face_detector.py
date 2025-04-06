#!/usr/bin/env python3
"""
Test script for the face detector component.
This script tests whether the OpenCV-based face detection is working correctly.
"""

import os
import cv2
import argparse
from utils.face_detector import FaceDetector

def test_face_detector(image_path, save_result=True):
    """
    Test the face detector on an image
    Args:
        image_path: Path to the test image
        save_result: Whether to save the result image
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return False

    # Create face detector
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"Error creating face detector: {e}")
        return False

    # Detect faces
    try:
        faces = detector.detect_faces(image)
        print(f"Detected {len(faces)} faces in the image")

        # Draw bounding boxes
        result_image = detector.draw_faces(image, faces)

        # Display the result
        cv2.imshow("Face Detection Test", result_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the result
        if save_result:
            base_name = os.path.basename(image_path)
            result_path = f"test_result_{base_name}"
            cv2.imwrite(result_path, result_image)
            print(f"Result saved to {result_path}")

        return True

    except Exception as e:
        print(f"Error during face detection: {e}")
        return False

def main():
    """Main function for the face detector test"""
    parser = argparse.ArgumentParser(description="Test the face detector component")
    parser.add_argument("--image", required=True, help="Path to the test image")
    parser.add_argument("--no-save", action="store_true", help="Don't save the result image")

    args = parser.parse_args()

    test_face_detector(args.image, not args.no_save)

if __name__ == "__main__":
    main()
