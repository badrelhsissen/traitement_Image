import cv2
import numpy as np

# Partie 2 : Détection d'Objets et Suivi

# Exercice 3 : Détection avancée de visages avec CNN
# Exercice 4 : Suivi d'objets en temps réel avec OpenCV

# Charger le modèle pré-entraîné de détection de visages
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Charger le modèle de détection de genre et d'estimation de l'âge
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")

# Les labels pour le genre et l'âge
GENDER_LIST = ["Homme", "Femme"]
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Dimensions des images d'entrée pour les réseaux
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Variables globales pour la sélection en temps réel
initBB = None  # Bounding box initiale (pour le suivi)
selecting_object = False  # Indicateur de sélection
tracker = None  # Le tracker
current_gender = None  # Genre actuel détecté
current_age = None  # Âge actuel détecté

# Fonction pour dessiner le rectangle de sélection en temps réel
def click_and_draw(event, x, y, flags, param):
    global initBB, selecting_object, tracker, frame
    
    # Début de la sélection de l'objet
    if event == cv2.EVENT_LBUTTONDOWN:
        initBB = (x, y, 0, 0)  # Initialiser la sélection
        selecting_object = True

    # Ajustement de la taille de la sélection
    elif event == cv2.EVENT_MOUSEMOVE and selecting_object:
        initBB = (initBB[0], initBB[1], x - initBB[0], y - initBB[1])

    # Fin de la sélection
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_object = False
        # Créer un tracker une fois la sélection terminée
        tracker = cv2.legacy.TrackerKCF_create()
        tracker.init(frame, initBB)

# Fonction pour détecter si la zone sélectionnée est un visage
def is_face_selected(face):
    blob = cv2.dnn.blobFromImage(cv2.resize(face, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    # Vérifier s'il y a un visage avec une confiance suffisante
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            return True
    return False

# Fonction pour détecter le genre et l'âge dans la zone sélectionnée
def detect_gender_age(face):
    blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Détection du genre
    gender_net.setInput(blob_face)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Estimation de l'âge
    age_net.setInput(blob_face)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    return gender, age

# Fonction pour suivre l'objet en temps réel et afficher le genre et l'âge
def track_object_in_realtime(video_capture):
    global initBB, tracker, selecting_object, frame, current_gender, current_age

    # Définir le callback pour la souris
    cv2.namedWindow("Selection et Suivi d'objet")
    cv2.setMouseCallback("Selection et Suivi d'objet", click_and_draw)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Si un objet a été sélectionné, appliquer le suivi
        if tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extraire la zone du visage pour la détection de genre et d'âge
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    # Vérifier si la zone sélectionnée est bien un visage
                    if is_face_selected(face):
                        current_gender, current_age = detect_gender_age(face)

                        # Afficher le genre et l'âge au-dessus de la zone suivie
                        label = f"{current_gender}, {current_age}"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    else:
                        # Si ce n'est pas un visage, ne rien afficher
                        current_gender, current_age = None, None
            else:
                cv2.putText(frame, "Objet perdu", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        # Si un rectangle est en cours de sélection, le dessiner
        if selecting_object and initBB is not None:
            (x, y, w, h) = [int(v) for v in initBB]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Afficher la vidéo avec suivi/rectangle en temps réel
        cv2.imshow("Selection et Suivi d'objet", frame)

        # Appuyer sur 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Capturer la vidéo à partir de la webcam
video_capture = cv2.VideoCapture(0)

# Appeler la fonction de suivi d'objet en temps réel
track_object_in_realtime(video_capture)
