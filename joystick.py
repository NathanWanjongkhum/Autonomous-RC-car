# This runs on your computer (not the Pi)
import pygame
import paramiko
import time
import math

# SSH connection info
PI_HOST = '10.19.4.242'
PI_USER = 'apple'
PI_PASS = '1234'

# Set up SSH and start motorcontrol.py
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS)
stdin, stdout, stderr = ssh.exec_command('python3 Documents/cooler_rc_Car/Autonomous-RC-car/motorcontrol.py', get_pty=True)

def send_command(cmd):
    stdin.write(cmd + '\n')
    stdin.flush()

# Pygame setup
pygame.init()
size = 400
screen = pygame.display.set_mode((size, size))
pygame.display.set_caption("Virtual Joystick")

center = (size // 2, size // 2)
radius = 120
stick_radius = 30
dragging = False
stick_pos = center

last_cmd = None

def get_direction(dx, dy):
    angle = math.atan2(-dy, dx)
    dist = math.hypot(dx, dy)
    if dist < 30:
        return 'stop'
    if -math.pi/4 < angle <= math.pi/4:
        return 'd'
    elif math.pi/4 < angle <= 3*math.pi/4:
        return 'w'
    elif -3*math.pi/4 < angle <= -math.pi/4:
        return 's'
    else:
        return 'a'

pressed_keys = set()

try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if math.hypot(event.pos[0]-center[0], event.pos[1]-center[1]) < radius:
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
                stick_pos = center
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - center[0]
                dy = event.pos[1] - center[1]
                dist = min(math.hypot(dx, dy), radius)
                angle = math.atan2(dy, dx)
                stick_pos = (
                    int(center[0] + dist * math.cos(angle)),
                    int(center[1] + dist * math.sin(angle))
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    pressed_keys.add('w')
                elif event.key == pygame.K_a:
                    pressed_keys.add('a')
                elif event.key == pygame.K_s:
                    pressed_keys.add('s')
                elif event.key == pygame.K_d:
                    pressed_keys.add('d')
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    pressed_keys.discard('w')
                elif event.key == pygame.K_a:
                    pressed_keys.discard('a')
                elif event.key == pygame.K_s:
                    pressed_keys.discard('s')
                elif event.key == pygame.K_d:
                    pressed_keys.discard('d')
                # If no keys are pressed, stop
                if not pressed_keys:
                    send_command('stop')
                    last_cmd = 'stop'

        # Handle keyboard hold
        key_priority = ['w', 'a', 's', 'd']
        key_cmd = None
        for k in key_priority:
            if k in pressed_keys:
                key_cmd = k
                break

        # If using keyboard, override joystick
        if key_cmd:
            if key_cmd != last_cmd:
                send_command(key_cmd)
                last_cmd = key_cmd
        else:
            # Joystick logic
            if not dragging:
                stick_pos = center

            dx = stick_pos[0] - center[0]
            dy = stick_pos[1] - center[1]
            cmd = get_direction(dx, dy)
            if cmd != last_cmd:
                send_command(cmd)
                last_cmd = cmd

        # Draw joystick
        screen.fill((30, 30, 30))
        pygame.draw.circle(screen, (100, 100, 100), center, radius, 5)
        pygame.draw.circle(screen, (0, 120, 255), stick_pos, stick_radius)
        pygame.display.flip()
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    try:
        if stdin and not stdin.closed:
            send_command('stopall')
            stdin.close()
    except Exception as e:
        print(f"Could not send stopall or close stdin: {e}")
    ssh.close()
    pygame.quit()