# This runs on your computer (not the Pi)
import pygame
import paramiko
import time
import math

# SSH connection info
PI_HOST = '10.19.4.242'
PI_USER = 'apple'
PI_PASS = '1234'
AUTONOMOUS_CMD = 'python3 Documents/RC_CAR/Autonomous-RC-car/planning/run_all.py'

# SSH setup
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS)

stdin = None
autonomous_proc = None
mode = 'manual'  # or 'autonomous'

def start_manual():
    global stdin, autonomous_proc, mode
    if autonomous_proc:
        autonomous_proc.channel.close()
        autonomous_proc = None
    stdin, stdout, stderr = ssh.exec_command('python3 Documents/RC_CAR/Autonomous-RC-car/control/motorcontrol.py', get_pty=True)
    mode = 'manual'
    print("Switched to MANUAL mode.")

def start_autonomous():
    global stdin, autonomous_proc, mode
    if stdin and not stdin.closed:
        stdin.write('stopall\n')
        stdin.flush()
        stdin.close()
        stdin = None
    autonomous_proc = ssh.exec_command(AUTONOMOUS_CMD, get_pty=True)[0]
    mode = 'autonomous'
    print("Switched to AUTONOMOUS mode.")

def stop_all():
    global stdin, autonomous_proc
    try:
        if stdin and not stdin.closed:
            stdin.write('stopall\n')
            stdin.flush()
            stdin.close()
            stdin = None
    except Exception as e:
        print(f"[DEBUG] Error stopping manual: {e}")
    try:
        if autonomous_proc:
            autonomous_proc.channel.close()
            autonomous_proc = None
    except Exception as e:
        print(f"[DEBUG] Error stopping autonomous: {e}")

def send_command(cmd):
    if mode == 'manual' and stdin and not stdin.closed:
        try:
            stdin.write(cmd + '\n')
            stdin.flush()
        except Exception as e:
            print(f"[DEBUG] Could not send command '{cmd}': {e}")

# Pygame setup
pygame.init()
size = 400
screen = pygame.display.set_mode((size, size))
pygame.display.set_caption("RC Car Joystick/Menu")

center = (size // 2, size // 2)
radius = 120
stick_radius = 30
dragging = False
stick_pos = center

last_cmd = None
pressed_keys = set()

# --- Button definitions ---
BUTTONS = [
    {"label": "Manual",     "rect": pygame.Rect(20, 320, 100, 40), "action": start_manual},
    {"label": "Autonomous", "rect": pygame.Rect(140, 320, 120, 40), "action": start_autonomous},
    {"label": "Stop",       "rect": pygame.Rect(280, 320, 80, 40), "action": stop_all},
    {"label": "Quit",       "rect": pygame.Rect(280, 20, 80, 40),  "action": None},  # handled in loop
]

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

def draw_menu():
    font = pygame.font.SysFont(None, 28)
    menu_lines = [
        f"Current Mode: {mode.upper()}"
    ]
    for i, line in enumerate(menu_lines):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, 10 + i * 30))
    # Draw mode indicator
    color = (0, 200, 0) if mode == 'autonomous' else (0, 120, 255)
    pygame.draw.circle(screen, color, (370, 50), 15)
    mode_text = font.render(mode.upper(), True, color)
    screen.blit(mode_text, (320, 70))
    # Draw buttons
    for btn in BUTTONS:
        pygame.draw.rect(screen, (70, 70, 70), btn["rect"])
        pygame.draw.rect(screen, (200, 200, 200), btn["rect"], 2)
        text = font.render(btn["label"], True, (255, 255, 255))
        text_rect = text.get_rect(center=btn["rect"].center)
        screen.blit(text, text_rect)

try:
    start_manual()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    start_manual()
                elif event.key == pygame.K_a:
                    start_autonomous()
                elif event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_w:
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
                if not pressed_keys and mode == 'manual':
                    send_command('stop')
                    last_cmd = 'stop'
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for btn in BUTTONS:
                    if btn["rect"].collidepoint(mx, my):
                        if btn["label"] == "Quit":
                            running = False
                        elif btn["action"]:
                            btn["action"]()
                if math.hypot(mx-center[0], my-center[1]) < radius:
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

        # Handle keyboard hold
        key_priority = ['w', 'a', 's', 'd']
        key_cmd = None
        for k in key_priority:
            if k in pressed_keys:
                key_cmd = k
                break

        if mode == 'manual':
            if key_cmd:
                if key_cmd != last_cmd:
                    send_command(key_cmd)
                    last_cmd = key_cmd
            else:
                if not dragging:
                    stick_pos = center
                dx = stick_pos[0] - center[0]
                dy = stick_pos[1] - center[1]
                cmd = get_direction(dx, dy)
                if cmd != last_cmd:
                    send_command(cmd)
                    last_cmd = cmd

        # Draw joystick and menu
        screen.fill((30, 30, 30))
        pygame.draw.circle(screen, (100, 100, 100), center, radius, 5)
        pygame.draw.circle(screen, (0, 120, 255), stick_pos, stick_radius)
        draw_menu()
        pygame.display.flip()
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    stop_all()
    ssh.close()
    pygame.quit()