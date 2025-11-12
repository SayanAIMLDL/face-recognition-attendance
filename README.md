# Face Recognition Attendance System

A robust, real-time attendance system built with Python, OpenCV, and the `face_recognition` library. The application identifies known individuals via a webcam and automatically logs their attendance into daily Excel spreadsheets, providing a modern and efficient solution for tracking presence in various environments.

## Key Features

-   **Real-Time Recognition:** Identifies multiple known and unknown faces from a live webcam feed.
-   **Automated Logging:** Automatically records the first appearance of a recognized person for the day.
-   **Date-wise Reports:** Generates and updates daily attendance reports in `.xlsx` (Excel) format.
-   **Advanced Enrollment:** A guided, interactive script for enrolling new users that captures multiple head poses (center, left, right, up, down) to improve recognition accuracy.
-   **Data Integrity:** Prevents duplicate attendance entries if the program is run multiple times on the same day.
-   **Intelligent & Efficient:**
    -   Processes video frames efficiently to reduce CPU load.
    -   The enrollment script validates that a face is clearly detected before saving an image.
-   **Clean User Interface:** Provides clear on-screen visual feedback with colored boxes (green for known, red for unknown) and name labels.

## Project Structure
