import cv2
import os
import time
import face_recognition

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"

# Define the sequence of poses required from the user
POSES = [
    {"name": "Center", "count": 2},
    {"name": "Look Left", "count": 1},
    {"name": "Look Right", "count": 1},
    {"name": "Look Up", "count": 1},
    {"name": "Look Down", "count": 1}
]

def enroll_person_advanced():
    """
    Guides a user through an advanced enrollment process with pose guidance
    and face detection validation.
    """
    # 1. Get the person's name (which will be their unique ID)
    person_name = input("Enter the name/ID for the person to enroll (e.g., 'john_doe'): ")
    if not person_name or any(c in r'<>:"/\|?*' for c in person_name):
        print("Invalid name. Please use only letters, numbers, underscores, and dashes. Aborting.")
        return

    person_path = os.path.join(KNOWN_FACES_DIR, person_name)

    # 2. Create a directory for the person's images, handling existing users
    if os.path.exists(person_path):
        print(f"Warning: A person with the name '{person_name}' already exists.")
        overwrite = input("Do you want to overwrite their images? (y/n): ").lower()
        if overwrite != 'y':
            print("Enrollment cancelled.")
            return
    else:
        os.makedirs(person_path)
    
    print(f"Directory for {person_name} is ready at {person_path}")

    # 3. Initialize Webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Advanced Enrollment Started ---")
    
    total_snapshots_taken = 0
    
    # Loop through each required pose
    for pose in POSES:
        pose_name = pose["name"]
        snapshots_to_take_for_pose = pose["count"]
        snapshots_taken_for_pose = 0

        print(f"\nNext Pose: Please **{pose_name}** and hold still.")
        
        # Give user time to get into position
        time.sleep(2.5)

        last_capture_time = time.time()

        while snapshots_taken_for_pose < snapshots_to_take_for_pose:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            display_frame = frame.copy()
            
            # Requirement #2: Check for a clear face before taking a snapshot
            face_locations = face_recognition.face_locations(frame)

            # Display instructions and status on the screen
            instruction_text = f"Pose: {pose_name} ({snapshots_taken_for_pose}/{snapshots_to_take_for_pose})"
            cv2.putText(display_frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(face_locations) == 1:
                # Exactly one face is detected, good to capture
                cv2.putText(display_frame, "Face Detected: OK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Requirement #1: Slow down snapshots
                if time.time() - last_capture_time >= 2.0: # 2-second delay
                    snapshot_path = os.path.join(person_path, f"{pose_name.lower().replace(' ', '_')}_{int(time.time())}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    print(f"  > Snapshot {snapshots_taken_for_pose + 1} for pose '{pose_name}' saved.")
                    
                    snapshots_taken_for_pose += 1
                    total_snapshots_taken += 1
                    last_capture_time = time.time()
            
            elif len(face_locations) > 1:
                cv2.putText(display_frame, "Error: Multiple Faces Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Searching for face...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.imshow('Advanced Enrollment - Press "Q" to Cancel', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Enrollment cancelled by user.")
                video_capture.release()
                cv2.destroyAllWindows()
                return

    # 4. Final Completion Message
    print("\n--- Enrollment Complete! ---")
    
    # Requirement #4: Show completion message on screen
    ret, frame = video_capture.read() # Get one last frame to draw on
    if ret:
        completion_text = "Process Completed!"
        # Get text size to center it
        text_size = cv2.getTextSize(completion_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(frame, completion_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('Advanced Enrollment - Press "Q" to Cancel', frame)
        print("Showing completion screen. Press 'q' to exit.")
        cv2.waitKey(0) # Wait indefinitely until a key is pressed

    # 5. Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    print(f"\nEnrollment process for '{person_name}' finished.")
    print(f"{total_snapshots_taken} total images were saved.")

if __name__ == "__main__":
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    enroll_person_advanced()