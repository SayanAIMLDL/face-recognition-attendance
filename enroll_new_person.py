import cv2
import os
import time

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
SNAPSHOTS_TO_TAKE = 5  # Number of snapshots to take for each person

def enroll_person():
    """
    Guides an administrator through enrolling a new person via the webcam.
    """
    # 1. Get the person's name (which will be their unique ID)
    person_name = input("Enter the name/ID for the person to enroll (e.g., 'john_doe'): ")
    if not person_name or any(c in r'<>:"/\|?*' for c in person_name):
        print("Invalid name. Please use only letters, numbers, underscores, and dashes. Aborting.")
        return

    person_path = os.path.join(KNOWN_FACES_DIR, person_name)

    # 2. Create a directory for the person's images
    if os.path.exists(person_path):
        print(f"Warning: A person with the name '{person_name}' already exists.")
        overwrite = input("Do you want to overwrite their images? (y/n): ").lower()
        if overwrite != 'y':
            print("Enrollment cancelled.")
            return
    else:
        os.makedirs(person_path)
        print(f"Directory created for {person_name} at {person_path}")

    # 3. Initialize Webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting webcam for enrollment.")
    print(f"Please look at the camera. We will take {SNAPSHOTS_TO_TAKE} snapshots.")
    print("The process will start in 3 seconds. Get ready...")
    time.sleep(3)

    snapshots_taken = 0
    while snapshots_taken < SNAPSHOTS_TO_TAKE:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Display feedback on the screen
        text = f"Taking snapshot {snapshots_taken + 1}/{SNAPSHOTS_TO_TAKE}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Enrollment - Press "Q" to Cancel', frame)
        
        # Save a snapshot
        snapshot_path = os.path.join(person_path, f"image_{int(time.time())}.jpg")
        cv2.imwrite(snapshot_path, frame)
        print(f"Saved snapshot {snapshots_taken + 1}/{SNAPSHOTS_TO_TAKE} at {snapshot_path}")
        snapshots_taken += 1
        
        # Wait for 1 second between snapshots to allow for slight pose changes
        time.sleep(1)

        # Allow quitting with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Enrollment cancelled by user.")
            break

    # 4. Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    print(f"\nEnrollment process for '{person_name}' finished.")
    print(f"{snapshots_taken} images were saved in the '{person_path}' folder.")

if __name__ == "__main__":
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    enroll_person()