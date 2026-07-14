from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

cascade = BASE_DIR / "data" / "Cascades" / "haarcascade_frontalface_default.xml"


detector = cv2.CascadeClassifier(str(cascade))

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Não foi possível abrir a câmera.")

while True:
    ok, frame = camera.read()

    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = detector.detectMultiScale(gray)

    for x, y, w, h in deteccoes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
