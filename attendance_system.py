import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime

# --- Configuration & Setup ---
KNOWN_FACES_DIR = "known_faces"
REPORTS_DIR = "attendance_reports"
# Process every Nth frame to save CPU, increase for better performance
FRAME_PROCESSING_INTERVAL = 5 

def load_known_faces():
    """
    Loads face encodings and names from the known_faces directory.
    Only loads one image per person to be efficient.
    
    Returns:
        tuple: (known_face_encodings, known_face_names)
    """
    known_face_encodings = []
    known_face_names = []
    print("Loading known faces from database...")

    if not os.path.exists(KNOWN_FACES_DIR) or not os.listdir(KNOWN_FACES_DIR):
        print("Warning: 'known_faces' directory is empty or not found.")
        print("Please enroll people using the 'enroll_new_person.py' script first.")
        return known_face_encodings, known_face_names

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    # Add the first found encoding to our list
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"Loaded sample ID for: {name}")
                    # We only need one sample per person for recognition
                    break 
                else:
                    print(f"Warning: No face detected in {image_path}. Skipping.")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    return known_face_encodings, known_face_names

def update_attendance_sheet(recognized_names):
    """
    Updates the Excel sheet for the current day with new attendees.
    Avoids duplicate entries if the program is run multiple times a day.
    
    Args:
        recognized_names (set): A set of names recognized in the current session.
    """
    if not recognized_names:
        return # Nothing to update

    today_str = datetime.now().strftime('%Y-%m-%d')
    excel_path = os.path.join(REPORTS_DIR, f"Attendance_{today_str}.xlsx")
    
    # Read existing data or create a new DataFrame
    try:
        df = pd.read_excel(excel_path)
        existing_names = set(df['Name'])
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Timestamp'])
        existing_names = set()

    new_attendees = []
    for name in recognized_names:
        if name not in existing_names:
            timestamp = datetime.now().strftime('%H:%M:%S')
            new_attendees.append({'Name': name, 'Timestamp': timestamp})
            print(f"Logging new attendance for {name} at {timestamp}")

    if new_attendees:
        new_df = pd.DataFrame(new_attendees)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(excel_path, index=False)
        print(f"Attendance sheet updated at {excel_path}")

def main():
    """
    Main function to run the real-time face recognition and attendance system.
    """
    known_face_encodings, known_face_names = load_known_faces()
    
    if not known_face_encodings:
        print("\nNo known faces loaded. Exiting application.")
        return

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting daily attendance system. Press 'Q' to quit.")
    
    # Store names recognized in this specific run of the program
    session_recognized_names = set() 
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Can't receive frame from webcam. Exiting.")
            break
            
        frame_count += 1
        # Process only every Nth frame to save resources
        if frame_count % FRAME_PROCESSING_INTERVAL == 0:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert BGR (OpenCV format) to RGB (face_recognition format)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if face_distances.size > 0:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        
                        if name not in session_recognized_names:
                            print(f"Recognized {name}. Adding to session log.")
                            session_recognized_names.add(name)

        # Drawing boxes and names on the frame (this part runs every frame for smooth video)
        # Note: This part can be simplified or removed if visual feedback is not needed
        # For now, we'll keep it simple and just show the raw frame
        cv2.imshow('Attendance System - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'Q' pressed. Shutting down...")
            break

    # --- Cleanup and Save ---
    video_capture.release()
    cv2.destroyAllWindows()
    
    print("\nSession ended. Finalizing attendance sheet...")
    update_attendance_sheet(session_recognized_names)
    print("Program exited successfully.")

if __name__ == "__main__":
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    main()