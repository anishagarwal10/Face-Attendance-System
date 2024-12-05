import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

video_capture = None
running = False

def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]
    else:
        print(f"No face detected in {image_path}")
        return None


known_face_encodings = [
    get_face_encoding("pic/anni.jpg"),
    get_face_encoding("pic/aniket.jpg"),
    get_face_encoding("pic/satendra.jpg"),
    get_face_encoding("pic/saurav.PNG"),
    get_face_encoding("pic/vishnu.jpg")
]
known_face_encodings = [enc for enc in known_face_encodings if enc is not None]

known_face_names = [
    "Anish Agarwal", "Aniket Goyal", "Satendra", "Saurav Chaudhary", "Vishnu Singh"
]

attendance_logged = {name: False for name in known_face_names}
current_date = datetime.now().strftime("%Y-%m-%d")

with open(f"{current_date}.csv", "w", newline="") as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])

def start_recognition():
    global video_capture, running
    video_capture = cv2.VideoCapture(0)
    running = True
    recognize_faces()

def stop_recognition():
    global video_capture, running
    running = False
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Stopped", "Face Recognition Stopped")

def recognize_faces():
    global video_capture, running

    if not running:
        return

    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if not attendance_logged[name]:
                attendance_logged[name] = True
                with open(f"{current_date}.csv", "a", newline="") as f:
                    lnwriter = csv.writer(f)
                    lnwriter.writerow([name, current_time])
                messagebox.showinfo("Attendance", f"Marked: {name} at {current_time}")

            cv2.putText(frame, f"{name} Present", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_recognition()
    else:
        root.after(10, recognize_faces)

def view_attendance():
    try:
        with open(f"{current_date}.csv", "r") as f:
            data = f.read()
        messagebox.showinfo("Attendance Log", data)
    except FileNotFoundError:
        messagebox.showerror("Error", "No attendance file found!")


root = tk.Tk()
root.title("Face Attendance System")

tk.Button(root, text="Start Recognition", command=start_recognition, width=20).pack(pady=10)
tk.Button(root, text="Stop Recognition", command=stop_recognition, width=20).pack(pady=10)
tk.Button(root, text="View Attendance", command=view_attendance, width=20).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", stop_recognition)
root.mainloop()
