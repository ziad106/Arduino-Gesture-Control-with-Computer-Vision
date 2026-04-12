import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam NOT accessible")
    exit(1)

print("✓ Webcam accessible. Displaying for 10 seconds...")
print("Press 'q' to quit early.")

frame_count = 0
while frame_count < 300:  # ~10 seconds at 30fps
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    frame_count += 1

    # Add text
    cv2.putText(frame, f"Frame: {frame_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✓ Webcam working! Captured {frame_count} frames")
