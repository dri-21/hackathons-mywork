from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import numpy as np
import time

def send_velocity_to_mole(v, w):
    print(f"Linear velocity: {v:.2f}, Angular velocity: {w:.2f}")

print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.getObject('sim')
vision_sensor = sim.getObject('/Hawk/visionSensor')

sim.startSimulation()
time.sleep(1)
print("Simulation started")

running = True

# -------- STATE VARIABLES (CRITICAL) --------
turning = False
turn_end_time = 0
TURN_DURATION = 0.8   # seconds (THIS FIXES CORNERS)

while running:
    try:
        img, resolution = sim.getVisionSensorImg(vision_sensor)
        if img is None or len(img) == 0:
            continue

        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape(resolution[1], resolution[0], 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        height, width, _ = img.shape
        now = time.time()

        # ---------- FRONT WALL (EDGE BASED) ----------
        front_roi = img[int(height*0.7):height, int(width*0.3):int(width*0.7)]
        front_edges = cv2.Canny(front_roi, 50, 150)
        front_wall = (cv2.countNonZero(front_edges) / front_edges.size) > 0.03

        # ---------- RIGHT WALL (EDGE BASED) ----------
        right_roi = img[int(height*0.7):height, int(width*0.6):width]
        right_edges = cv2.Canny(right_roi, 50, 150)
        right_wall = (cv2.countNonZero(right_edges) / right_edges.size) > 0.03

        # -------------------------------------------------
        # TURN COMMITMENT LOGIC (THIS FIXES EVERYTHING)
        # -------------------------------------------------
        if turning:
            if now < turn_end_time:
                # Continue the turn
                decision = "TURNING"
                linear_velocity = 0.2
                angular_velocity = current_turn_dir
                send_velocity_to_mole(linear_velocity, angular_velocity)
            else:
                turning = False
            continue

        # -------------------------------------------------
        # RIGHT-HAND WALL FOLLOWING (DECISION POINT)
        # -------------------------------------------------
        if not right_wall:
            decision = "TURN RIGHT"
            turning = True
            current_turn_dir = -0.6
            turn_end_time = now + TURN_DURATION

        elif not front_wall:
            decision = "FORWARD"
            linear_velocity = 0.5
            angular_velocity = 0.0
            send_velocity_to_mole(linear_velocity, angular_velocity)

        else:
            decision = "TURN LEFT"
            turning = True
            current_turn_dir = 0.6
            turn_end_time = now + TURN_DURATION

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------
        cv2.putText(img, decision, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Hawk Camera", img)
        cv2.imshow("Front Edges", front_edges)
        cv2.imshow("Right Edges", right_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    except Exception as e:
        print("Runtime error:", e)

print("Stopping simulation...")
sim.stopSimulation()
cv2.destroyAllWindows()
print("Exited cleanly")