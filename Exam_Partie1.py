import cv2
import numpy as np
import matplotlib.pyplot as plt


# Partie 1 : Traitement d'Images

#Exercice 1 : Transformation d'image avancée

# Chargement de l'image en haute résolution
image_path = "pic_high_def.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Chargement en niveaux de gris pour simplifier le traitement.

# Fonction pour redimensionner l'image
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Application du filtre Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # Dérivée horizontale
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Dérivée verticale
sobel_combined = cv2.magnitude(sobelx, sobely)  # Magnitude des gradients

# Sauvegarder l'image Sobel
cv2.imwrite("sobel_image.jpg", sobel_combined)

# Transformation de Fourier
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Magnitude du spectre de Fourier
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Manipulation du spectre (par exemple, masque ou filtrage)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols, 2), np.uint8)
r = 30  # rayon du masque pour le filtrage passe-bas
mask[crow - r:crow + r, ccol - r:ccol + r] = 0

# Application du masque et transformation inverse
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Sauvegarder l'image après transformation inverse
cv2.imwrite("fourier_inverse_image.jpg", img_back)

# Segmentation par seuillage adaptatif
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Sauvegarder l'image segmentée
cv2.imwrite("threshold_image.jpg", thresh)

# Définir la taille des fenêtres (quart de l'écran)
screen_width = 1920  # Exemple de résolution écran
screen_height = 1080  # Exemple de résolution écran
window_width = screen_width // 2
window_height = screen_height // 2

# Affichage des résultats
cv2.namedWindow("Image originale en gris", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image originale en gris", window_width, window_height)
cv2.imshow("Image originale en gris", resize_image(img, 50))  # 50% pour réduire l'image

cv2.namedWindow("Sobel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sobel", window_width, window_height)
cv2.imshow("Sobel", resize_image(sobel_combined, 50))

cv2.namedWindow("Spectre de Fourier", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Spectre de Fourier", window_width, window_height)
cv2.imshow("Spectre de Fourier", resize_image(magnitude_spectrum, 50))

cv2.namedWindow("Image apres Fourier inverse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image apres Fourier inverse", window_width, window_height)
cv2.imshow("Image apres Fourier inverse", resize_image(img_back, 50))

cv2.namedWindow("Seuillage adaptatif", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Seuillage adaptatif", window_width, window_height)
cv2.imshow("Seuillage adaptatif", resize_image(thresh, 50))


# Exercice 2 : Masquage et ROI (Region of Interest)

# Charger l'image
img2 = cv2.imread(image_path)

# Vérifier si l'image est correctement chargée
if img2 is None:
    raise ValueError("L'image n'a pas pu être chargée.")

# Sélectionner une région d'intérêt (ROI) manuellement pour le monument (en fonction de l'image, approximativement au centre)
# Ici, les coordonnées sont estimées en fonction de l'arc de triomphe visible.
x, y, w, h = 1400, 500, 2900, 2700  # Ajustement des dimensions selon l'image et le monument central
roi = img2[y:y+h, x:x+w]

# Appliquer une modification sur la ROI
roi_mod = cv2.convertScaleAbs(roi, alpha=1.2, beta=50)  # Augmentation de la luminosité

# Remettre la ROI modifiée dans l'image originale
img2[y:y+h, x:x+w] = roi_mod

# Appliquer un flou gaussien sur le reste de l'image (hors de la ROI)
mask = np.zeros_like(img2)
mask[y:y+h, x:x+w] = 1  # Mask pour la ROI

# Appliquer le flou gaussien seulement en dehors de la ROI
blurred_img2 = cv2.GaussianBlur(img2, (51, 51), 0)
img2_final = np.where(mask == 1, img2, blurred_img2)

# Affichage de l'image finale
cv2.namedWindow("Image avec flou", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image avec flou", window_width, window_height)
cv2.imshow("Image avec flou", resize_image(img2_final, 50))

# Sauvegarder l'image modifiée
output_path = "image_avec_flou.jpg"
cv2.imwrite(output_path, img2_final)

# Attente d'une touche pour fermer la fenêtre
cv2.waitKey(0)
cv2.destroyAllWindows()

