import cv2                                                                                                     # Import the opencv library

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')                                  # Load the Haar classifier for face detection
smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')                                               # Load the Haar classifier for smile detection

cap = cv2.VideoCapture(0)                                                                                      # Initialize the camera and associate it with a variable

while True:                                                                                                    # Loop that runs while the program is open
    
    ret, video = cap.read()                                                                                    # Start capturing a frame from the camera video
    video = cv2.flip(video, 1)                                                                                 # Flip the image horizontally

    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)                                                             # Convert the video capture from color to grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                                                        # Detect faces in the grayscale video

    for (x, y, w, h) in faces:                                                                                 # Loop to track the movements of the faces
        
        cv2.rectangle(video, (x, y), (x + w, y + h), (255, 255, 255), 2)                                       # Draw a white rectangle around the face

        roi_gray = gray[y:y + h, x:x + w]                                                                      # Define the region of interest in grayscale
        roi_color = video[y:y + h, x:x + w]                                                                    # Define the region of interest in color

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.6, minNeighbors=25, minSize=(25, 25))  # Detect smiles within the face region

        for (sx, sy, sw, sh) in smiles:                                                                        # Loop to track smile movements
            
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)                         # Draw a white rectangle around the smile
            
            cv2.putText(video, "Smile Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) # Draw yellow text indicating "Smile Detected"

    cv2.imshow('Face and Smile Recognition', video)                                                            # Display the image with the results

    key = cv2.waitKey(1) & 0xFF                                                                                # Define a variable for the ESC key
    
    if key == 27:                                                                                              # Conditional for pressing ESC
        break                                                                                                  # Exit the program

cap.release()                                                                                                  # Release resources
cv2.destroyAllWindows()                                                                                        # Close all windows



