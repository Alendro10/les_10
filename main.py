import cv2                    # OpenCV для роботи з відео та зображенням
import mediapipe as mp        # MediaPipe — бібліотека для відстеження рук
import numpy as np            # NumPy для числових обчислень (не використовується напряму тут, але може бути корисним)

# Ініціалізація модулів MediaPipe
mp_hands = mp.solutions.hands                    # Модуль для розпізнавання рук
mp_drawing = mp.solutions.drawing_utils          # Утиліти для візуалізації

# Відкриваємо камеру (0 — стандартна веб-камера)
cap = cv2.VideoCapture(0)

# Створюємо об'єкт Hands
with mp_hands.Hands(
    static_image_mode=False,          # Встановлено False, щоб обробляти відеопотік у реальному часі
    max_num_hands=2,                  # Максимум 2 руки на кадр
    min_detection_confidence=0.5,     # Мінімальний рівень впевненості для первинного розпізнавання руки
    min_tracking_confidence=0.5       # Мінімальна впевненість для подальшого відстеження
) as hands:

    while cap.isOpened():             # Поки камера працює
        ret, frame = cap.read()       # Зчитуємо кадр з відео
        if not ret:
            break                     # Якщо кадр не вдалося зчитати — виходимо з циклу

        # Перетворюємо зображення в RGB (MediaPipe очікує саме такий формат)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False             # Забороняємо запис у зображення для пришвидшення обробки
        results = hands.process(image)            # Обробка кадру — пошук рук

        # Повертаємо зображення назад у BGR (OpenCV використовує цей формат)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Якщо знайдено хоча б одну руку
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape             # Висота і ширина зображення

                # Отримуємо координати кожної точки руки (landmarks)
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                # Визначаємо індекси landmark'ів для кожного пальця
                fingers = {
                    'Thumb': [1, 2, 3, 4],         # Великий палець
                    'Index': [5, 6, 7, 8],         # Вказівний
                    'Middle': [9, 10, 11, 12],     # Середній
                    'Ring': [13, 14, 15, 16],      # Безіменний
                    'Pinky': [17, 18, 19, 20]      # Мізинець
                }

                # Колір для кожного пальця
                colors = {
                    'Thumb': (255, 0, 0),          # Синій
                    'Index': (0, 255, 0),          # Зелений
                    'Middle': (0, 0, 255),         # Червоний
                    'Ring': (255, 255, 0),         # Жовтий
                    'Pinky': (255, 0, 255)         # Фіолетовий
                }

                # Малюємо лінії між точками для кожного пальця
                for finger, idxs in fingers.items():
                    for i in range(len(idxs) - 1):
                        pt1 = points[idxs[i]]
                        pt2 = points[idxs[i + 1]]
                        cv2.line(image, pt1, pt2, colors[finger], 4)

                # Малюємо круг у точці зап’ястя (landmark №0)
                cv2.circle(image, points[0], 5, (200, 200, 200), -1)

        # Показ обробленого зображення у вікні
        cv2.imshow('Hand Finger Detection', image)

        # Вихід із циклу при натисканні клавіші 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Завершення роботи з камерою та закриття всіх вікон
cap.release()
cv2.destroyAllWindows()