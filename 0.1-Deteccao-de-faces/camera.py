from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

cascade = BASE_DIR / "data" / "Cascades" / "haarcascade_frontalface_default.xml"


detector = cv2.CascadeClassifier(str(cascade))

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Não foi possível abrir a câmera.")


def pixelizar(img, tamanho=16):
    pequeno = cv2.resize(img, (tamanho, tamanho), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(
        pequeno, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
    )


while True:
    ok, frame = camera.read()

    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = detector.detectMultiScale(gray)

    for x, y, w, h in deteccoes:
        # frame[y:y+h, x:x+w] = cv2.blur(
        #     frame[y:y+h, x:x+w], (30, 30)
        # )
        frame[y:y+h, x:x+w] = pixelizar(frame[y:y+h, x:x+w], tamanho=8)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
