import cv2
import numpy as np

# Fonction pour créer un masque de peau
def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Définir les plages de couleur de la peau en HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Créer le masque de la peau
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Appliquer une série de transformations pour améliorer la détection
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

# Fonction pour analyser les gestes (paume ouverte ou poing fermé)
def analyze_gesture(contour, frame):
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return "Poing ferme"

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Calculer la longueur entre les points
        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(far) - np.array(end))

        # Calculer l'angle pour distinguer les doigts de la main
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi

        # Si l'angle est petit, c'est un doigt
        if angle <= 90:
            finger_count += 1
            cv2.circle(frame, far, 5, [0, 255, 0], -1)

    # Si plus de 3 doigts sont visibles, la main est ouverte
    if finger_count >= 3:
        return "Paume ouverte"
    else:
        return "Poing ferme"

# Fonction pour détecter les mains et reconnaître les gestes
def detect_hands_and_recognize_gestures():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Appliquer le masque de peau
        mask = skin_mask(frame)

        # Trouver les contours sur le masque de peau
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Ajuster ce seuil pour ignorer les petits objets
                # Dessiner le contour de la main
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)

                # Analyser le geste
                gesture = analyze_gesture(contour, frame)

                # Obtenir les coordonnées du contour et afficher le geste
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Afficher la vidéo avec la détection de gestes
        cv2.imshow("Detection de gestes", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exécuter la fonction de détection
detect_hands_and_recognize_gestures()
