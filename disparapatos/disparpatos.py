#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DUCK HUNT - Edición Pygame
==========================

Requisitos:  pip install pygame
Ejecutar:    python duck_hunt.py

MODOS
  1. PC vs Usuario                  (mouse)
  2. PC vs Usuario 1 vs Usuario 2   (mouse + teclado)
  3. Usuario vs Usuario             (mouse + teclado)

CONTROLES
  Usuario / Usuario 1 : mouse para apuntar, clic izquierdo para disparar
  Usuario 2           : flechas (o WASD) para mover la mira, ESPACIO / ENTER para disparar
  P = pausa   M = silenciar   ESC = volver al menú

ASSETS (todos OPCIONALES: si falta alguno el juego usa gráficos/silencio de reemplazo)
  assets/
    images/
      duck.png            <- sprite del pato mirando a la DERECHA (se escala solo)
      crosshair.png       <- imagen de la mira (se tiñe con el color de cada jugador)
      bg_level_1.png ...  <- bg_level_10.png  (fondo de cada nivel, ideal 1024x640)
    sounds/
      shot.wav            <- efecto de disparo
      duck_fall.wav       <- pato abatido / cayendo
      level_up.wav        <- nivel completado
      music.mp3           <- música de fondo (en bucle)
"""

import math
import os
import random
import sys

import pygame

# ============================================================================
#  CONFIGURACIÓN GENERAL
# ============================================================================
WIDTH, HEIGHT = 1024, 640
FPS = 60
GROUND_H = 110                      # altura de la franja de suelo
SKY_BOTTOM = HEIGHT - GROUND_H      # límite inferior donde vuelan los patos
HUD_TOP = 44                        # altura de la barra superior
TOTAL_LEVELS = 10
SHOT_COOLDOWN = 0.28                # segundos entre disparos de un mismo jugador
KEY_CURSOR_SPEED = 480              # px/s de la mira del teclado
MUSIC_VOLUME = 0.35

# --- Rutas de recursos (PLACEHOLDERS: coloca aquí tus archivos) --------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
IMG_DIR = os.path.join(ASSETS_DIR, "images")
SND_DIR = os.path.join(ASSETS_DIR, "sounds")

SOUND_FILES = {                     # clave: (archivo, volumen)
    "shot": ("shot.wav", 0.6),            # efecto de disparo
    "fall": ("duck_fall.wav", 0.7),       # pato abatido
    "level_up": ("level_up.wav", 0.7),    # nivel superado
}
MUSIC_FILE = "music.mp3"                  # música de fondo
DUCK_IMAGE = "duck.png"                   # sprite del pato (mirando a la derecha)
CROSSHAIR_IMAGE = "crosshair.png"         # mira opcional
BACKGROUND_PATTERN = "bg_level_{n}.png"   # fondo por nivel (n = 1..10)

# --- Dificultades ------------------------------------------------------------
#  erratic: 0 = recto, 1 = giros suaves, 2 = zigzag errático
DIFFICULTIES = {
    "facil":   dict(label="Fácil",   duck_size=110, base_speed=110, erratic=0,
                    base_time=70, points=100, cpu_acc=0.30),
    "medio":   dict(label="Medio",   duck_size=80,  base_speed=180, erratic=1,
                    base_time=55, points=150, cpu_acc=0.40),
    "dificil": dict(label="Difícil", duck_size=55,  base_speed=270, erratic=2,
                    base_time=45, points=250, cpu_acc=0.50),
}


# --- Progresión por nivel ----------------------------------------------------
def level_speed_mult(level):
    """Cada nivel sube un 15 % la velocidad de los patos."""
    return 1.0 + 0.15 * (level - 1)


def level_quota(level):
    """Patos que hay que abatir (entre todos) para pasar de nivel: 5, 7, 9 ... 23."""
    return 5 + 2 * (level - 1)


def level_time(diff, level):
    """Tiempo límite del nivel (segundos)."""
    return diff["base_time"] + 2 * level_quota(level)


def level_max_ducks(level):
    """Patos simultáneos en pantalla."""
    return min(2 + (level + 1) // 2, 6)


# Colores de jugadores
RED = (230, 70, 70)
BLUE = (70, 150, 255)
GREEN = (90, 220, 120)
WHITE = (255, 255, 255)


# ============================================================================
#  GESTOR DE RECURSOS (sonido + imágenes, con reemplazos si faltan archivos)
# ============================================================================
class Assets:
    def __init__(self):
        self.sounds = {}
        self.audio_ok = False
        self.muted = False
        self._warned = set()
        self._duck_cache = {}
        self._crosshair = None
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 2, 512)
            self.audio_ok = True
        except pygame.error as exc:
            print(f"[assets] Audio no disponible: {exc}")
        if self.audio_ok:
            for key, (filename, volume) in SOUND_FILES.items():
                self._load_sound(key, filename, volume)
        self._load_crosshair()

    # ---- utilidades ----
    def _warn(self, path):
        if path not in self._warned:
            self._warned.add(path)
            print(f"[assets] Falta {os.path.relpath(path, BASE_DIR)} (se usará un reemplazo)")

    def _load_sound(self, key, filename, volume):
        path = os.path.join(SND_DIR, filename)
        if not os.path.exists(path):
            self._warn(path)
            return
        try:
            snd = pygame.mixer.Sound(path)
            snd.set_volume(volume)
            self.sounds[key] = snd
        except pygame.error as exc:
            print(f"[assets] No se pudo cargar {filename}: {exc}")

    def load_image(self, filename, size=None):
        path = os.path.join(IMG_DIR, filename)
        if not os.path.exists(path):
            return None
        try:
            img = pygame.image.load(path).convert_alpha()
            if size:
                img = pygame.transform.smoothscale(img, size)
            return img
        except pygame.error as exc:
            print(f"[assets] No se pudo cargar {filename}: {exc}")
            return None

    # ---- audio ----
    def play(self, key):
        if self.audio_ok and not self.muted and key in self.sounds:
            self.sounds[key].play()

    def play_music(self):
        if not self.audio_ok:
            return
        path = os.path.join(SND_DIR, MUSIC_FILE)
        if not os.path.exists(path):
            self._warn(path)
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(0 if self.muted else MUSIC_VOLUME)
            pygame.mixer.music.play(-1)  # bucle infinito
        except pygame.error as exc:
            print(f"[assets] No se pudo reproducir la música: {exc}")

    def toggle_mute(self):
        self.muted = not self.muted
        if self.audio_ok:
            try:
                pygame.mixer.music.set_volume(0 if self.muted else MUSIC_VOLUME)
            except pygame.error:
                pass

    # ---- mira ----
    def _load_crosshair(self):
        self._crosshair = self.load_image(CROSSHAIR_IMAGE, (48, 48))

    def crosshair(self, color):
        """Devuelve la mira (png teñida) o None si hay que dibujarla por código."""
        if self._crosshair is None:
            return None
        img = self._crosshair.copy()
        img.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        return img

    # ---- patos ----
    def duck_frames(self, size, variety):
        """(frames_derecha, frames_izquierda): [ala arriba, ala abajo, golpeado]."""
        key = (size, variety)
        if key not in self._duck_cache:
            w, h = size, int(size * 0.75)
            sprite = self.load_image(DUCK_IMAGE, (w, h))
            if sprite is not None:
                # Con un único png se simulan las poses. Si tienes una hoja de
                # sprites, sustituye estas 3 líneas por tus propios cuadros.
                squashed = pygame.transform.smoothscale(sprite, (w, int(h * 0.88)))
                canvas = pygame.Surface((w, h), pygame.SRCALPHA)
                canvas.blit(squashed, (0, h - squashed.get_height()))
                right = [sprite, canvas, pygame.transform.flip(sprite, False, True)]
            else:
                right = make_duck_frames(size, variety)
            left = [pygame.transform.flip(f, True, False) for f in right]
            self._duck_cache[key] = (right, left)
        return self._duck_cache[key]


DUCK_PALETTES = [  # (cuerpo, cabeza, cola/ala)
    ((150, 100, 55), (30, 140, 70), (100, 65, 35)),
    ((120, 120, 130), (40, 90, 160), (80, 80, 95)),
    ((200, 160, 90), (170, 60, 50), (140, 105, 55)),
]


def make_duck_frames(size, variety):
    """Genera un pato con formas geométricas (reemplazo si no hay duck.png)."""
    body_c, head_c, dark_c = DUCK_PALETTES[variety % len(DUCK_PALETTES)]
    w, h = size, int(size * 0.75)
    frames = []
    for pose in ("up", "down", "hit"):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        # cola
        pygame.draw.polygon(s, dark_c, [(0.04 * w, 0.38 * h), (0.24 * w, 0.46 * h), (0.2 * w, 0.72 * h)])
        # cuerpo
        pygame.draw.ellipse(s, body_c, (int(0.1 * w), int(0.38 * h), int(0.6 * w), int(0.5 * h)))
        # cuello + cabeza
        pygame.draw.rect(s, head_c, (int(0.62 * w), int(0.2 * h), int(0.12 * w), int(0.3 * h)))
        pygame.draw.circle(s, head_c, (int(0.78 * w), int(0.26 * h)), max(3, int(0.17 * w)))
        pygame.draw.rect(s, WHITE, (int(0.62 * w), int(0.42 * h), int(0.12 * w), max(2, int(0.05 * h))))
        # pico
        pygame.draw.polygon(s, (255, 170, 30), [(0.9 * w, 0.2 * h), (1.0 * w, 0.3 * h), (0.88 * w, 0.35 * h)])
        # ojo
        ex, ey = int(0.82 * w), int(0.22 * h)
        if pose == "hit":
            r = max(2, int(0.03 * w))
            pygame.draw.line(s, (0, 0, 0), (ex - r, ey - r), (ex + r, ey + r), 2)
            pygame.draw.line(s, (0, 0, 0), (ex - r, ey + r), (ex + r, ey - r), 2)
        else:
            pygame.draw.circle(s, (0, 0, 0), (ex, ey), max(2, int(0.03 * w)))
        # ala
        if pose == "up":
            pts = [(0.28 * w, 0.52 * h), (0.5 * w, 0.0), (0.62 * w, 0.52 * h)]
        elif pose == "down":
            pts = [(0.28 * w, 0.6 * h), (0.46 * w, 0.99 * h), (0.62 * w, 0.62 * h)]
        else:
            pts = [(0.25 * w, 0.55 * h), (0.5 * w, 0.2 * h), (0.65 * w, 0.58 * h)]
        pygame.draw.polygon(s, dark_c, pts)
        frames.append(s)
    return frames


# ============================================================================
#  FONDOS PROCEDURALES (10 paisajes, uno por nivel)
#  Para usar imágenes propias: assets/images/bg_level_N.png
# ============================================================================
def vertical_gradient(surface, top, bottom, height):
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        pygame.draw.line(surface, color, (0, y), (WIDTH, y))


def alpha_circle(surf, color, center, radius, alpha):
    radius = int(radius)
    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
    surf.blit(s, (center[0] - radius, center[1] - radius))


def deco_sun(surf, rng, pos, r, color):
    for mult, a in ((3, 30), (2, 50), (1.4, 80)):
        alpha_circle(surf, color, pos, r * mult, a)
    pygame.draw.circle(surf, color, pos, r)


def deco_moon(surf, rng, pos, r, color=(240, 240, 215)):
    alpha_circle(surf, color, pos, r * 1.8, 30)
    pygame.draw.circle(surf, color, pos, r)
    for dx, dy, cr in ((-.3, -.2, .2), (.25, .15, .15), (-.1, .4, .12)):
        pygame.draw.circle(surf, (210, 210, 185), (int(pos[0] + dx * r), int(pos[1] + dy * r)), int(cr * r))


def deco_clouds(surf, rng, n=5, color=(255, 255, 255), ymin=60, ymax=230):
    for _ in range(n):
        x, y, k = rng.randint(0, WIDTH), rng.randint(ymin, ymax), rng.uniform(0.8, 1.5)
        for dx, dy, w, h in ((0, 0, 90, 38), (-30, 8, 70, 30), (35, 8, 75, 30), (10, -14, 60, 36)):
            pygame.draw.ellipse(surf, color, (x + dx * k, y + dy * k, w * k, h * k))


def deco_hills(surf, rng, color, base, amp):
    phase, f = rng.uniform(0, 6.28), rng.uniform(0.004, 0.008)
    pts = [(0, SKY_BOTTOM)]
    for x in range(0, WIDTH + 16, 16):
        y = base - amp * (0.5 + 0.35 * math.sin(x * f + phase) + 0.15 * math.sin(x * f * 2.3))
        pts.append((x, y))
    pts.append((WIDTH, SKY_BOTTOM))
    pygame.draw.polygon(surf, color, pts)


def deco_mountains(surf, rng, color, snow=None, n=6, hmin=140, hmax=260):
    for i in range(n):
        cx = i * WIDTH / (n - 1) + rng.randint(-40, 40)
        h = rng.randint(hmin, hmax)
        w = h * 1.5
        pygame.draw.polygon(surf, color, [(cx - w / 2, SKY_BOTTOM), (cx, SKY_BOTTOM - h), (cx + w / 2, SKY_BOTTOM)])
        if snow:
            pygame.draw.polygon(surf, snow, [(cx - w * 0.15, SKY_BOTTOM - h * 0.7), (cx, SKY_BOTTOM - h),
                                             (cx + w * 0.15, SKY_BOTTOM - h * 0.7), (cx, SKY_BOTTOM - h * 0.62)])


def deco_trees(surf, rng, n, trunk, leaf, hmin=80, hmax=150, style="pine"):
    for _ in range(n):
        x, h = rng.randint(0, WIDTH), rng.randint(hmin, hmax)
        base = SKY_BOTTOM + 10
        pygame.draw.rect(surf, trunk, (x - 5, base - h * 0.35, 10, h * 0.35))
        if style == "pine":
            for i in range(3):
                top = base - h * (0.35 + 0.3 * i + 0.3)
                bot = base - h * (0.2 + 0.28 * i)
                half = h * (0.32 - 0.07 * i)
                pygame.draw.polygon(surf, leaf, [(x - half, bot), (x, top), (x + half, bot)])
        else:
            pygame.draw.circle(surf, leaf, (x, int(base - h * 0.6)), int(h * 0.32))
            pygame.draw.circle(surf, leaf, (x - int(h * 0.2), int(base - h * 0.45)), int(h * 0.22))
            pygame.draw.circle(surf, leaf, (x + int(h * 0.2), int(base - h * 0.45)), int(h * 0.22))


def deco_cacti(surf, rng, n, color):
    for _ in range(n):
        x, h = rng.randint(20, WIDTH - 20), rng.randint(60, 110)
        base = SKY_BOTTOM + 14
        pygame.draw.rect(surf, color, (x - 8, base - h, 16, h), border_radius=8)
        pygame.draw.rect(surf, color, (x - 28, base - h * 0.65, 12, h * 0.3), border_radius=6)
        pygame.draw.rect(surf, color, (x - 28, base - h * 0.45, 24, 10), border_radius=5)
        pygame.draw.rect(surf, color, (x + 16, base - h * 0.75, 12, h * 0.3), border_radius=6)
        pygame.draw.rect(surf, color, (x + 4, base - h * 0.55, 24, 10), border_radius=5)


def deco_stars(surf, rng, n):
    for _ in range(n):
        x, y = rng.randint(0, WIDTH), rng.randint(0, int(SKY_BOTTOM * 0.75))
        c = rng.randint(170, 255)
        pygame.draw.circle(surf, (c, c, c), (x, y), rng.choice((1, 1, 2)))


def deco_volcano(surf, rng, cx, h, color, lava):
    alpha_circle(surf, lava, (cx, SKY_BOTTOM - h), 130, 55)
    pygame.draw.polygon(surf, color, [(cx - h * 0.9, SKY_BOTTOM), (cx - h * 0.15, SKY_BOTTOM - h),
                                      (cx + h * 0.15, SKY_BOTTOM - h), (cx + h * 0.9, SKY_BOTTOM)])
    pygame.draw.polygon(surf, lava, [(cx - h * 0.15, SKY_BOTTOM - h), (cx + h * 0.15, SKY_BOTTOM - h),
                                     (cx + h * 0.08, SKY_BOTTOM - h * 0.8), (cx - h * 0.1, SKY_BOTTOM - h * 0.85)])
    pygame.draw.line(surf, lava, (cx, SKY_BOTTOM - h * 0.8), (cx - h * 0.25, SKY_BOTTOM - h * 0.3), 5)
    pygame.draw.line(surf, lava, (cx + 10, SKY_BOTTOM - h * 0.8), (cx + h * 0.3, SKY_BOTTOM - h * 0.2), 4)


def deco_aurora(surf, rng, colors):
    layer = pygame.Surface((WIDTH, SKY_BOTTOM), pygame.SRCALPHA)
    for i, col in enumerate(colors):
        base, phase = 110 + i * 70, rng.uniform(0, 6.28)
        top = [(x, base + 35 * math.sin(x * 0.006 + phase)) for x in range(0, WIDTH + 20, 20)]
        bottom = [(x, y + 80) for x, y in reversed(top)]
        pygame.draw.polygon(layer, (*col, 70), top + bottom)
    surf.blit(layer, (0, 0))


def deco_reeds(surf, rng, n, color):
    for _ in range(n):
        x, h = rng.randint(0, WIDTH), rng.randint(40, 90)
        base = SKY_BOTTOM + 25
        pygame.draw.line(surf, color, (x, base), (x + rng.randint(-6, 6), base - h), 3)
        pygame.draw.ellipse(surf, color, (x - 4, base - h - 12, 9, 20))


def deco_tufts(surf, rng, color):
    for _ in range(90):
        x, y = rng.randint(0, WIDTH), rng.randint(SKY_BOTTOM + 14, HEIGHT - 4)
        pygame.draw.line(surf, color, (x, y), (x + rng.randint(-3, 3), y - rng.randint(5, 11)), 2)


DECOR = {
    "sun": deco_sun, "moon": deco_moon, "clouds": deco_clouds, "hills": deco_hills,
    "mountains": deco_mountains, "trees": deco_trees, "cacti": deco_cacti, "stars": deco_stars,
    "volcano": deco_volcano, "aurora": deco_aurora, "reeds": deco_reeds, "tufts": deco_tufts,
}

# Un tema por nivel: cielo (arriba, abajo), color de suelo, decorado de fondo y de frente
THEMES = [
    dict(name="Amanecer en la pradera", sky=((255, 170, 110), (150, 210, 240)), ground=(80, 165, 70),
         back=[("sun", dict(pos=(780, SKY_BOTTOM - 60), r=55, color=(255, 230, 150))),
               ("clouds", dict(n=4, color=(255, 225, 210))),
               ("hills", dict(color=(110, 170, 100), base=SKY_BOTTOM - 10, amp=70))],
         front=[("tufts", dict(color=(50, 120, 45)))]),
    dict(name="Mediodía soleado", sky=((70, 160, 255), (185, 228, 255)), ground=(50, 170, 70),
         back=[("sun", dict(pos=(160, 110), r=50, color=(255, 240, 120))),
               ("clouds", dict(n=6)),
               ("hills", dict(color=(80, 175, 90), base=SKY_BOTTOM - 20, amp=90))],
         front=[("tufts", dict(color=(30, 120, 40)))]),
    dict(name="Bosque de pinos", sky=((120, 190, 230), (205, 232, 240)), ground=(30, 110, 50),
         back=[("clouds", dict(n=4)),
               ("hills", dict(color=(60, 130, 90), base=SKY_BOTTOM - 40, amp=90))],
         front=[("trees", dict(n=16, trunk=(90, 60, 30), leaf=(25, 95, 50))),
                ("tufts", dict(color=(20, 80, 35)))]),
    dict(name="Atardecer rojizo", sky=((255, 110, 60), (255, 205, 130)), ground=(95, 70, 45),
         back=[("sun", dict(pos=(512, SKY_BOTTOM - 30), r=85, color=(255, 215, 100))),
               ("mountains", dict(color=(120, 60, 60), n=6)),
               ],
         front=[("tufts", dict(color=(60, 40, 25)))]),
    dict(name="Desierto", sky=((255, 215, 150), (255, 245, 205)), ground=(222, 182, 105),
         back=[("sun", dict(pos=(850, 100), r=48, color=(255, 250, 190))),
               ("hills", dict(color=(235, 200, 130), base=SKY_BOTTOM - 20, amp=80))],
         front=[("cacti", dict(n=7, color=(60, 130, 60))), ("tufts", dict(color=(180, 140, 70)))]),
    dict(name="Montañas nevadas", sky=((150, 190, 230), (232, 242, 252)), ground=(235, 240, 246),
         back=[("clouds", dict(n=4)),
               ("mountains", dict(color=(110, 130, 165), snow=(250, 252, 255), n=6, hmin=170, hmax=300))],
         front=[("trees", dict(n=8, trunk=(80, 60, 40), leaf=(40, 100, 80), hmin=70, hmax=110)),
                ("tufts", dict(color=(190, 205, 220)))]),
    dict(name="Pantano nublado", sky=((105, 125, 118), (175, 195, 175)), ground=(62, 84, 52),
         back=[("clouds", dict(n=8, color=(150, 160, 155), ymin=40, ymax=200)),
               ("hills", dict(color=(70, 95, 70), base=SKY_BOTTOM - 20, amp=60))],
         front=[("reeds", dict(n=45, color=(45, 70, 40))),
                ("trees", dict(n=6, trunk=(60, 45, 30), leaf=(50, 80, 50), style="round", hmin=90, hmax=140))]),
    dict(name="Noche de luna", sky=((8, 10, 40), (42, 42, 105)), ground=(22, 52, 32),
         back=[("stars", dict(n=130)), ("moon", dict(pos=(840, 110), r=42)),
               ("hills", dict(color=(20, 35, 70), base=SKY_BOTTOM - 20, amp=80))],
         front=[("trees", dict(n=12, trunk=(30, 25, 20), leaf=(15, 45, 35))),
                ("tufts", dict(color=(15, 40, 25)))]),
    dict(name="Volcán en erupción", sky=((50, 6, 10), (232, 92, 30)), ground=(48, 32, 32),
         back=[("clouds", dict(n=5, color=(75, 40, 40), ymin=40, ymax=180)),
               ("volcano", dict(cx=512, h=300, color=(60, 40, 40), lava=(255, 120, 30)))],
         front=[("tufts", dict(color=(90, 50, 40)))]),
    dict(name="Aurora boreal", sky=((4, 6, 32), (16, 56, 72)), ground=(30, 30, 58),
         back=[("stars", dict(n=170)),
               ("aurora", dict(colors=[(60, 255, 150), (90, 160, 255), (180, 90, 255)])),
               ("mountains", dict(color=(15, 25, 50), snow=(120, 150, 190), n=7, hmin=80, hmax=170))],
         front=[("tufts", dict(color=(60, 60, 110)))]),
]


def build_background(level, assets=None):
    """Fondo del nivel: usa bg_level_N.png si existe; si no, lo dibuja por código."""
    if assets is not None:
        img = assets.load_image(BACKGROUND_PATTERN.format(n=level), (WIDTH, HEIGHT))
        if img is not None:
            return img.convert()
    theme = THEMES[(level - 1) % len(THEMES)]
    rng = random.Random(level * 97)
    surf = pygame.Surface((WIDTH, HEIGHT))
    vertical_gradient(surf, theme["sky"][0], theme["sky"][1], SKY_BOTTOM)
    for name, kw in theme["back"]:
        DECOR[name](surf, rng, **kw)
    pygame.draw.rect(surf, theme["ground"], (0, SKY_BOTTOM, WIDTH, GROUND_H))
    darker = tuple(max(0, c - 35) for c in theme["ground"])
    pygame.draw.rect(surf, darker, (0, SKY_BOTTOM, WIDTH, 12))
    for name, kw in theme["front"]:
        DECOR[name](surf, rng, **kw)
    return surf


# ============================================================================
#  PATO
# ============================================================================
class Duck:
    def __init__(self, diff, level, frames):
        self.w = diff["duck_size"]
        self.h = int(self.w * 0.75)
        self.frames_r, self.frames_l = frames
        self.erratic = diff["erratic"]
        speed = diff["base_speed"] * level_speed_mult(level) * random.uniform(0.9, 1.1)
        self.speed = speed
        self.pos = pygame.Vector2(random.uniform(self.w, WIDTH - self.w), SKY_BOTTOM - self.h / 2)
        angle = math.radians(random.uniform(215, 325))      # hacia arriba
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.state = "fly"        # fly -> hit -> fall -> gone
        self.t = random.uniform(0, 10)
        self.phase = random.uniform(0, 6.28)
        self.timer = 0.0
        self.steer_timer = random.uniform(0.3, 1.0)
        self.angle = 0.0
        self.facing_right = self.vel.x >= 0
        self.owner = None

    # ---- lógica ----
    def _steer(self, dt):
        """Cambios de dirección según la dificultad."""
        if self.erratic == 0:
            return
        self.steer_timer -= dt
        if self.steer_timer > 0:
            return
        if self.erratic == 1:       # Medio: giros suaves y poco frecuentes
            self.vel.rotate_ip(random.uniform(-35, 35))
            self.steer_timer = random.uniform(1.2, 2.5)
        else:                       # Difícil: zigzag brusco y frecuente
            self.vel.rotate_ip(random.choice((-1, 1)) * random.uniform(50, 110))
            self.steer_timer = random.uniform(0.25, 0.7)

    def update(self, dt):
        self.t += dt
        if self.state == "fly":
            self._steer(dt)
            self.pos += self.vel * dt
            if self.erratic == 2:   # ondulación extra en zigzag
                self.pos.y += math.sin(self.t * 12 + self.phase) * 60 * dt
            # rebote en los bordes del cielo
            if self.pos.x < self.w / 2:
                self.pos.x, self.vel.x = self.w / 2, abs(self.vel.x)
            elif self.pos.x > WIDTH - self.w / 2:
                self.pos.x, self.vel.x = WIDTH - self.w / 2, -abs(self.vel.x)
            if self.pos.y < HUD_TOP + self.h / 2:
                self.pos.y, self.vel.y = HUD_TOP + self.h / 2, abs(self.vel.y)
            elif self.pos.y > SKY_BOTTOM - self.h / 2:
                self.pos.y, self.vel.y = SKY_BOTTOM - self.h / 2, -abs(self.vel.y)
            if abs(self.vel.x) > 1:
                self.facing_right = self.vel.x > 0
        elif self.state == "hit":
            self.timer -= dt
            if self.timer <= 0:
                self.state = "fall"
                self.vel = pygame.Vector2(0, 80)
        elif self.state == "fall":
            self.vel.y += 900 * dt
            self.pos += self.vel * dt
            self.angle += 540 * dt
            if self.pos.y > SKY_BOTTOM + 15:
                self.state = "gone"

    def hit_test(self, point):
        rect = pygame.Rect(0, 0, int(self.w * 0.85), int(self.h * 0.9))
        rect.center = (int(self.pos.x), int(self.pos.y))
        return rect.collidepoint(point)

    def shoot_down(self, owner):
        self.state = "hit"
        self.timer = 0.25
        self.owner = owner
        self.vel = pygame.Vector2(0, 0)

    def draw(self, surf):
        frames = self.frames_r if self.facing_right else self.frames_l
        if self.state == "fly":
            img = frames[int(self.t * 10) % 2]
        elif self.state == "hit":
            img = frames[2]
        else:
            img = pygame.transform.rotate(frames[2], self.angle)
        surf.blit(img, img.get_rect(center=(int(self.pos.x), int(self.pos.y))))


# ============================================================================
#  JUGADOR (mouse / teclado / PC automática)
# ============================================================================
class Player:
    def __init__(self, name, kind, color):
        self.name = name
        self.kind = kind            # "mouse" | "keys" | "cpu"
        self.color = color
        self.pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        if kind == "mouse":
            self.pos.update(pygame.mouse.get_pos())
        self.score = 0
        self.shots = 0
        self.hits = 0
        self.cooldown = 0.0
        self.flash = 0.0
        # IA de la PC
        self.ai_state = "idle"
        self.ai_timer = 1.0
        self.target = None

    @property
    def hint(self):
        return {"mouse": "Mouse · clic", "keys": "Flechas · ESPACIO", "cpu": "Automática"}[self.kind]

    def update(self, dt, game):
        self.cooldown = max(0.0, self.cooldown - dt)
        self.flash = max(0.0, self.flash - dt)
        if self.kind == "mouse":
            self.pos.update(pygame.mouse.get_pos())
        elif self.kind == "keys":
            k = pygame.key.get_pressed()
            dx = (k[pygame.K_RIGHT] or k[pygame.K_d]) - (k[pygame.K_LEFT] or k[pygame.K_a])
            dy = (k[pygame.K_DOWN] or k[pygame.K_s]) - (k[pygame.K_UP] or k[pygame.K_w])
            move = pygame.Vector2(dx, dy)
            if move.length_squared() > 0:
                self.pos += move.normalize() * KEY_CURSOR_SPEED * dt
        else:
            self._ai(dt, game)
        self.pos.x = max(0, min(WIDTH, self.pos.x))
        self.pos.y = max(0, min(HEIGHT, self.pos.y))

    def _ai(self, dt, game):
        """PC: elige un pato, mueve su mira hacia él y dispara con cierta precisión."""
        if self.ai_state == "idle":
            self.ai_timer -= dt
            if self.ai_timer <= 0:
                targets = [d for d in game.ducks if d.state == "fly"]
                if targets:
                    self.target = random.choice(targets)
                    self.ai_state = "aim"
                    self.ai_timer = max(0.45, random.uniform(0.9, 1.8) - 0.06 * game.level)
                else:
                    self.ai_timer = 0.3
            return
        t = self.target
        if t is None or t.state != "fly":
            self.ai_state, self.ai_timer = "idle", random.uniform(0.2, 0.6)
            return
        lead = t.pos + t.vel * 0.12
        self.pos += (lead - self.pos) * min(1.0, 6 * dt)
        self.ai_timer -= dt
        if self.ai_timer <= 0:
            accuracy = game.diff["cpu_acc"] + 0.015 * (game.level - 1)
            if random.random() < accuracy:
                self.pos.update(t.pos)                         # tiro certero
            else:                                              # tiro fallado
                off = pygame.Vector2(random.uniform(60, 140), 0).rotate(random.uniform(0, 360))
                self.pos.update(t.pos + off)
            self.shoot(game)
            self.ai_state, self.ai_timer = "idle", random.uniform(0.3, 0.9)

    def shoot(self, game):
        if self.cooldown > 0 or game.state != "playing":
            return
        self.cooldown = SHOT_COOLDOWN
        self.flash = 0.12
        self.shots += 1
        game.assets.play("shot")
        for duck in reversed(game.ducks):
            if duck.state == "fly" and duck.hit_test(self.pos):
                duck.shoot_down(self)
                pts = game.diff["points"] + 10 * (game.level - 1)
                self.score += pts
                self.hits += 1
                game.on_duck_hit(self, duck, pts)
                return


# ============================================================================
#  BOTÓN DE MENÚ
# ============================================================================
class Button:
    def __init__(self, label, center, callback, sub=None, size=(560, 78)):
        self.label, self.sub, self.callback = label, sub, callback
        self.rect = pygame.Rect(0, 0, *size)
        self.rect.center = center

    def draw(self, surf, font, small, mouse):
        hover = self.rect.collidepoint(mouse)
        bg = (255, 205, 70) if hover else (30, 45, 90)
        fg = (30, 30, 30) if hover else WHITE
        pygame.draw.rect(surf, bg, self.rect, border_radius=14)
        pygame.draw.rect(surf, WHITE, self.rect, 3, border_radius=14)
        y = self.rect.centery - (12 if self.sub else 0)
        img = font.render(self.label, True, fg)
        surf.blit(img, img.get_rect(center=(self.rect.centerx, y)))
        if self.sub:
            img = small.render(self.sub, True, (60, 60, 60) if hover else (190, 205, 240))
            surf.blit(img, img.get_rect(center=(self.rect.centerx, self.rect.centery + 20)))


# ============================================================================
#  JUEGO
# ============================================================================
class Game:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Duck Hunt - Pygame")
        self.clock = pygame.time.Clock()
        self.assets = Assets()
        self.running = True

        def font(sz, bold=True):
            return pygame.font.SysFont("arial", sz, bold=bold)
        self.f_title, self.f_big = font(78), font(54)
        self.f_med, self.f_small = font(30), font(18, False)
        self.f_hud = font(24)

        self.mode = None
        self.diff_key = "medio"
        self.players = []
        self.ducks = []
        self.popups = []
        self.level = 1
        self.quota = self.downed = 0
        self.time_left = 0.0
        self.state_timer = 0.0
        self.spawn_timer = 0.0
        self.max_ducks = 3
        self.end_reason = ""
        self.bg_cache = {}
        self.bg = self.get_background(1)

        # patos decorativos del menú
        self.menu_ducks = [self.make_duck(DIFFICULTIES["medio"], 3) for _ in range(4)]

        self.state = "menu_mode"
        self.buttons = []
        self.build_mode_menu()
        self.assets.play_music()

    # ---------- utilidades ----------
    @property
    def diff(self):
        return DIFFICULTIES[self.diff_key]

    def get_background(self, level):
        if level not in self.bg_cache:
            self.bg_cache[level] = build_background(level, self.assets)
        return self.bg_cache[level]

    def make_duck(self, diff, level):
        frames = self.assets.duck_frames(diff["duck_size"], random.randrange(3))
        return Duck(diff, level, frames)

    def text(self, s, font, color, pos, anchor="center", shadow=True):
        img = font.render(s, True, color)
        rect = img.get_rect(**{anchor: pos})
        if shadow:
            self.screen.blit(font.render(s, True, (0, 0, 0)), rect.move(2, 2))
        self.screen.blit(img, rect)
        return rect

    # ---------- menús ----------
    def build_mode_menu(self):
        cx = WIDTH // 2
        self.state = "menu_mode"
        self.buttons = [
            Button("1 · PC vs Usuario", (cx, 270), lambda: self.choose_mode(1),
                   "Tú con el mouse contra la PC automática"),
            Button("2 · PC vs Usuario 1 vs Usuario 2", (cx, 365), lambda: self.choose_mode(2),
                   "Mouse (Usuario 1) y teclado (Usuario 2) contra la PC"),
            Button("3 · Usuario vs Usuario", (cx, 460), lambda: self.choose_mode(3),
                   "Duelo local: mouse contra teclado"),
        ]

    def build_diff_menu(self):
        cx = WIDTH // 2
        self.state = "menu_diff"
        self.buttons = [
            Button("1 · Fácil", (cx, 250), lambda: self.choose_diff("facil"),
                   "Patos grandes, lentos y tiempo amplio"),
            Button("2 · Medio", (cx, 340), lambda: self.choose_diff("medio"),
                   "Patos de tamaño estándar, velocidad moderada"),
            Button("3 · Difícil", (cx, 430), lambda: self.choose_diff("dificil"),
                   "Patos pequeños, rápidos y en zigzag"),
            Button("← Volver (ESC)", (cx, 520), self.build_mode_menu, size=(300, 56)),
        ]

    def choose_mode(self, mode):
        self.mode = mode
        self.build_diff_menu()

    def choose_diff(self, key):
        self.diff_key = key
        self.start_game()

    def go_menu(self):
        pygame.mouse.set_visible(True)
        self.build_mode_menu()

    # ---------- flujo de partida ----------
    def start_game(self):
        if self.mode == 1:
            self.players = [Player("PC", "cpu", RED), Player("Usuario", "mouse", BLUE)]
        elif self.mode == 2:
            self.players = [Player("PC", "cpu", RED), Player("Usuario 1", "mouse", BLUE),
                            Player("Usuario 2", "keys", GREEN)]
        else:
            self.players = [Player("Usuario 1", "mouse", BLUE), Player("Usuario 2", "keys", GREEN)]
        self.level = 1
        pygame.mouse.set_visible(False)
        self.start_level()

    def start_level(self):
        self.ducks, self.popups = [], []
        self.quota = level_quota(self.level)
        self.downed = 0
        self.time_left = level_time(self.diff, self.level)
        self.max_ducks = level_max_ducks(self.level)
        self.spawn_timer = 0.0
        self.bg = self.get_background(self.level)
        for p in self.players:
            p.cooldown, p.ai_state, p.ai_timer, p.target = 0.0, "idle", 1.0, None
        self.state, self.state_timer = "intro", 2.2

    def finish_level(self):
        for d in self.ducks:
            if d.state == "fly":
                d.state = "gone"
        self.assets.play("level_up")
        self.state, self.state_timer = "level_clear", 3.0

    def end_game(self, reason):
        self.end_reason = reason
        self.state = "game_over"
        pygame.mouse.set_visible(True)
        cx = WIDTH // 2
        self.buttons = [
            Button("Jugar de nuevo (R)", (cx - 190, 560), self.start_game, size=(340, 60)),
            Button("Menú principal (ESC)", (cx + 190, 560), self.go_menu, size=(340, 60)),
        ]

    def on_duck_hit(self, player, duck, pts):
        self.downed += 1
        self.assets.play("fall")
        self.popups.append(dict(text=f"+{pts}", pos=pygame.Vector2(duck.pos), life=1.0, color=player.color))

    def result_lines(self):
        ranking = sorted(self.players, key=lambda p: p.score, reverse=True)
        if len(ranking) > 1 and ranking[0].score == ranking[1].score:
            headline = "¡Empate!"
        elif self.mode == 1:
            headline = "¡Ganaste!" if ranking[0].kind != "cpu" else "Gana la PC"
        else:
            headline = f"Ganador: {ranking[0].name}"
        return headline, ranking

    # ---------- eventos ----------
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            self.running = False
            return
        if e.type == pygame.KEYDOWN and e.key == pygame.K_m:
            self.assets.toggle_mute()

        if self.state in ("menu_mode", "menu_diff"):
            self._menu_event(e)
        elif self.state == "game_over":
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    self.start_game()
                elif e.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                    self.go_menu()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for b in self.buttons:
                    if b.rect.collidepoint(e.pos):
                        b.callback()
                        break
        else:  # intro / playing / paused / level_clear
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.go_menu()
                elif e.key == pygame.K_p and self.state in ("playing", "paused"):
                    self.state = "paused" if self.state == "playing" else "playing"
                elif e.key in (pygame.K_SPACE, pygame.K_RETURN):
                    for p in self.players:
                        if p.kind == "keys":
                            p.shoot(self)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for p in self.players:
                    if p.kind == "mouse":
                        p.pos.update(e.pos)
                        p.shoot(self)

    def _menu_event(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_KP1, pygame.K_KP2, pygame.K_KP3):
                idx = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
                       pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2}[e.key]
                self.buttons[idx].callback()
            elif e.key == pygame.K_ESCAPE:
                if self.state == "menu_diff":
                    self.build_mode_menu()
                else:
                    self.running = False
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for b in self.buttons:
                if b.rect.collidepoint(e.pos):
                    b.callback()
                    break

    # ---------- actualización ----------
    def update(self, dt):
        for p in self.popups:
            p["life"] -= dt
            p["pos"].y -= 40 * dt
        self.popups = [p for p in self.popups if p["life"] > 0]

        if self.state in ("menu_mode", "menu_diff"):
            for d in self.menu_ducks:
                d.update(dt)
        elif self.state == "intro":
            for p in self.players:
                p.update(dt, self)
            self.state_timer -= dt
            if self.state_timer <= 0:
                self.state = "playing"
        elif self.state == "playing":
            self.update_play(dt)
        elif self.state == "level_clear":
            for p in self.players:
                p.update(dt, self)
            for d in self.ducks:
                d.update(dt)
            self.state_timer -= dt
            if self.state_timer <= 0:
                if self.level >= TOTAL_LEVELS:
                    self.end_game("¡Completaste los 10 niveles!")
                else:
                    self.level += 1
                    self.start_level()

    def update_play(self, dt):
        self.time_left -= dt
        for p in self.players:
            p.update(dt, self)
        for d in self.ducks:
            d.update(dt)
        self.ducks = [d for d in self.ducks if d.state != "gone"]

        # reponer patos (sin sobrepasar los que faltan por abatir)
        flying = sum(1 for d in self.ducks if d.state == "fly")
        wanted = min(self.max_ducks, self.quota - self.downed)
        self.spawn_timer -= dt
        if flying < wanted and self.spawn_timer <= 0:
            self.ducks.append(self.make_duck(self.diff, self.level))
            self.spawn_timer = 0.5

        if self.downed >= self.quota:
            self.finish_level()
        elif self.time_left <= 0:
            self.end_game("¡Se acabó el tiempo!")

    # ---------- dibujo ----------
    def draw_crosshair(self, p):
        pos = (int(p.pos.x), int(p.pos.y))
        img = self.assets.crosshair(p.color)
        if img is not None:
            self.screen.blit(img, img.get_rect(center=pos))
        else:
            for col, wd in ((WHITE, 5), (p.color, 3)):
                pygame.draw.circle(self.screen, col, pos, 18, wd)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    pygame.draw.line(self.screen, col, (pos[0] + dx * 10, pos[1] + dy * 10),
                                     (pos[0] + dx * 28, pos[1] + dy * 28), wd)
            pygame.draw.circle(self.screen, p.color, pos, 3)
        if p.flash > 0:
            pygame.draw.circle(self.screen, WHITE, pos, 26 + int(p.flash * 120), 2)
        tag = self.f_small.render(p.name, True, p.color)
        self.screen.blit(tag, tag.get_rect(center=(pos[0], pos[1] - 38)))

    @staticmethod
    def fmt_time(t):
        t = max(0, math.ceil(t))
        return f"{t // 60}:{t % 60:02d}"

    def draw_hud(self):
        bar = pygame.Surface((WIDTH, HUD_TOP), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 150))
        self.screen.blit(bar, (0, 0))
        self.text(f"Nivel {self.level}/{TOTAL_LEVELS} · {self.diff['label']}", self.f_hud, WHITE,
                  (14, HUD_TOP // 2), "midleft", False)
        # barra de progreso de patos
        bw, bx = 240, WIDTH // 2 - 120
        pygame.draw.rect(self.screen, (60, 60, 60), (bx, 10, bw, 24), border_radius=8)
        fill = int(bw * min(1, self.downed / self.quota))
        pygame.draw.rect(self.screen, (255, 200, 60), (bx, 10, fill, 24), border_radius=8)
        pygame.draw.rect(self.screen, WHITE, (bx, 10, bw, 24), 2, border_radius=8)
        self.text(f"Patos {self.downed}/{self.quota}", self.f_small, (20, 20, 20), (WIDTH // 2, HUD_TOP // 2), shadow=False)
        color = (255, 90, 90) if self.time_left < 10 else WHITE
        self.text(f"Tiempo {self.fmt_time(self.time_left)}", self.f_hud, color, (WIDTH - 14, HUD_TOP // 2), "midright", False)

        # marcadores abajo
        n = len(self.players)
        for i, p in enumerate(self.players):
            cx = int(WIDTH * (i + 1) / (n + 1))
            panel = pygame.Surface((300, 78), pygame.SRCALPHA)
            pygame.draw.rect(panel, (0, 0, 0, 150), panel.get_rect(), border_radius=12)
            pygame.draw.rect(panel, (*p.color, 255), panel.get_rect(), 3, border_radius=12)
            self.screen.blit(panel, (cx - 150, HEIGHT - 94))
            self.text(p.name, self.f_hud, p.color, (cx - 138, HEIGHT - 80), "midleft", False)
            self.text(str(p.score), self.f_big, WHITE, (cx + 138, HEIGHT - 62), "midright", False)
            self.text(p.hint, self.f_small, (200, 200, 200), (cx - 138, HEIGHT - 36), "midleft", False)

    def dim(self, alpha=140):
        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ov.fill((0, 0, 0, alpha))
        self.screen.blit(ov, (0, 0))

    def draw(self):
        s = self.screen
        if self.state in ("menu_mode", "menu_diff"):
            s.blit(self.get_background(1), (0, 0))
            for d in self.menu_ducks:
                d.draw(s)
            self.text("DUCK HUNT", self.f_title, (255, 220, 70), (WIDTH // 2, 100))
            sub = "Elige el modo de juego" if self.state == "menu_mode" else "Elige la dificultad"
            self.text(sub, self.f_med, WHITE, (WIDTH // 2, 170 if self.state == "menu_mode" else 170))
            mouse = pygame.mouse.get_pos()
            for b in self.buttons:
                b.draw(s, self.f_med, self.f_small, mouse)
            self.text("Mouse: apuntar + clic   ·   Usuario 2: flechas/WASD + ESPACIO   ·   P pausa   ·   M silencio",
                      self.f_small, WHITE, (WIDTH // 2, HEIGHT - 30))
            return

        s.blit(self.bg, (0, 0))
        if self.state == "game_over":
            self.dim(170)
            headline, ranking = self.result_lines()
            self.text(self.end_reason, self.f_big, (255, 220, 70), (WIDTH // 2, 100))
            self.text(headline, self.f_big, WHITE, (WIDTH // 2, 180))
            self.text(f"Nivel alcanzado: {self.level}/{TOTAL_LEVELS} · {self.diff['label']}",
                      self.f_med, (200, 220, 255), (WIDTH // 2, 245))
            for i, p in enumerate(ranking):
                acc = int(100 * p.hits / p.shots) if p.shots else 0
                self.text(f"{i + 1}. {p.name}: {p.score} pts   ({p.hits} patos, {acc}% precisión)",
                          self.f_med, p.color, (WIDTH // 2, 320 + i * 50))
            mouse = pygame.mouse.get_pos()
            for b in self.buttons:
                b.draw(s, self.f_med, self.f_small, mouse)
            return

        for d in self.ducks:
            d.draw(s)
        for p in self.popups:
            self.text(p["text"], self.f_med, p["color"], (int(p["pos"].x), int(p["pos"].y)))
        self.draw_hud()

        if self.state == "intro":
            self.dim(110)
            theme = THEMES[(self.level - 1) % len(THEMES)]
            self.text(f"NIVEL {self.level}", self.f_title, (255, 220, 70), (WIDTH // 2, 220))
            self.text(theme["name"], self.f_med, WHITE, (WIDTH // 2, 290))
            self.text(f"Objetivo: {self.quota} patos · Tiempo: {self.fmt_time(self.time_left)}",
                      self.f_med, WHITE, (WIDTH // 2, 340))
        elif self.state == "level_clear":
            self.dim(110)
            self.text(f"¡Nivel {self.level} completado!", self.f_big, (255, 220, 70), (WIDTH // 2, 200))
            for i, p in enumerate(sorted(self.players, key=lambda q: q.score, reverse=True)):
                self.text(f"{p.name}: {p.score}", self.f_med, p.color, (WIDTH // 2, 270 + i * 44))
        elif self.state == "paused":
            self.dim(150)
            self.text("PAUSA", self.f_title, WHITE, (WIDTH // 2, 270))
            self.text("P para continuar · ESC para salir al menú", self.f_med, WHITE, (WIDTH // 2, 340))

        for p in self.players:
            self.draw_crosshair(p)

    # ---------- bucle principal ----------
    def run(self):
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            for e in pygame.event.get():
                self.handle_event(e)
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()


def main():
    Game().run()


if __name__ == "__main__":
    main()
