import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

particles = []  
max_particles = 100  
fireball_radius = 40  

lightning_particles = []  
max_lightning_particles = 40 
lightning_speed = 20
lightning_color = (0, 255, 255)  

colors = [(255, 69, 0), (255, 140, 0), (255, 255, 0), (255, 255, 255)] 

class Particle:
    def __init__(self, x, y, color=lightning_color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-lightning_speed, lightning_speed)
        self.vy = random.uniform(-lightning_speed, lightning_speed)
        self.life = random.randint(20, 50) 
        self.color = color
        self.size = random.randint(1, 3)

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.size = max(self.size - 0.05, 1)  


def is_snapping(landmarks):
    thumb_tip = landmarks.landmark[4]
    middle_tip = landmarks.landmark[12]
    distance = np.sqrt((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2)
    return distance < 0.04  

def is_clapping(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.04  

def generate_lightning_particles(palm_x, palm_y):
    for _ in range(max_lightning_particles):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(lightning_speed, lightning_speed * 1.5)
        length = random.uniform(50, 100)
        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed
        lightning_particles.append(Particle(palm_x, palm_y, lightning_color))

def trigger_blast(palm_x, palm_y):
    for _ in range(100): 
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(5, 15)
        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed
        lightning_particles.append(Particle(palm_x, palm_y, (0, 255, 255)))  

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    overlay = frame.copy()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

            h, w, _ = frame.shape
            palm_x = int(hand_landmarks.landmark[9].x * w)
            palm_y = int(hand_landmarks.landmark[9].y * h)

            if len(particles) < max_particles:
                particles.append(Particle(palm_x, palm_y))
            cv2.circle(overlay, (palm_x, palm_y), fireball_radius, (255, 69, 0), -1)
            cv2.circle(overlay, (palm_x, palm_y), fireball_radius + 10, (255, 255, 0), 2)
            cv2.circle(overlay, (palm_x, palm_y), fireball_radius + 20, (255, 255, 255), 1)

            if is_snapping(hand_landmarks):
                generate_lightning_particles(palm_x, palm_y)

            if is_clapping(hand_landmarks):
                trigger_blast(palm_x, palm_y)

    for lightning in lightning_particles[:]:
        lightning.move()
        if lightning.life <= 0:
            lightning_particles.remove(lightning)
        else:
            cv2.circle(overlay, (int(lightning.x), int(lightning.y)), int(lightning.size), lightning.color, -1)

    for particle in particles[:]:
        particle.move()
        if particle.life <= 0:
            particles.remove(particle)
        else:
            cv2.circle(overlay, (int(particle.x), int(particle.y)), int(particle.size), particle.color, -1)

    alpha = 0.7 
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow("Lightning Shot Effect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit command received. Exiting loop.")
        break
cap.release()
cv2.destroyAllWindows()
