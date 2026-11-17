import cv2

image_input = cv2.imread(r"dataset\organico\000105.jpg")
bgr_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
hsv_output = cv2.cvtColor(bgr_rgb, cv2.COLOR_RGB2HSV)

cv2.imshow("imagem_hsv2.jpg", hsv_output)

