import math
import os
import random
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import pygame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# ----------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------
ANCHO, ALTO = 1100, 650
FPS = 60
GRAVEDAD = 0.65
FUERZA_SALTO = -13.5
VEL_JUGADOR = 5.0
SUELO_Y = 610
PUNTOS_FRUTA = 1
PUNTOS_BANDERA = 5
ARCHIVO_RANKING = "ranking_colombia.csv"
MAX_NOMBRE = 12
# Dificultades: velocidad de la PC, probabilidad/duración de sus "dudas" y tiempo del nivel
DIFICULTADES = [
    dict(nombre="Fácil",   desc="PC lenta · 90 s",  vel=3.0, p_pausa=0.020, pausa=(40, 90), tiempo=90),
    dict(nombre="Medio",   desc="PC normal · 80 s", vel=3.8, p_pausa=0.008, pausa=(20, 50), tiempo=80),
    dict(nombre="Difícil", desc="PC veloz · 70 s",  vel=4.6, p_pausa=0.002, pausa=(10, 20), tiempo=70),
]
# ----------------------------------------------------------------
# DATOS (pandas): catálogo de animales
# ----------------------------------------------------------------
ANIMALES = pd.DataFrame([
    dict(clave="jaguar", nombre="Jaguar", region="Amazonía",
         dato="Es el felino más grande de América y vive en las selvas colombianas."),
    dict(clave="oso", nombre="Oso de anteojos", region="Páramos y Andes",
         dato="Es el único oso de Sudamérica y ayuda a dispersar semillas en los bosques andinos."),
    dict(clave="perezoso", nombre="Perezoso", region="Costa Caribe",
         dato="Se mueve muy despacio y baja del árbol solo cada cierto tiempo para hacer sus necesidades."),
    dict(clave="chiguiro", nombre="Chigüiro", region="Llanos Orientales",
         dato="Es el roedor más grande del mundo y vive cerca de ríos y humedales."),
    dict(clave="condor", nombre="Cóndor andino", region="Nevados Andinos",
         dato="Es una de las aves voladoras más grandes del mundo y aparece en el escudo de Colombia."),
    dict(clave="rana", nombre="Rana dorada", region="Pacífico",
         dato="La rana dorada venenosa vive en el Pacífico colombiano y es de las más tóxicas del mundo."),
]).set_index("clave", drop=False)
# Colores pastel de cada animal
COLORES = {
    "jaguar":   dict(cuerpo=(255, 218, 160), acento=(224, 168, 110)),
    "oso":      dict(cuerpo=(212, 178, 150), acento=(160, 125, 100)),
    "perezoso": dict(cuerpo=(232, 214, 184), acento=(170, 140, 115)),
    "chiguiro": dict(cuerpo=(214, 184, 140), acento=(170, 138, 100)),
    "condor":   dict(cuerpo=(196, 204, 220), acento=(140, 150, 175)),
    "rana":     dict(cuerpo=(255, 238, 140), acento=(168, 220, 150)),
}
# ----------------------------------------------------------------
# NIVELES
#   ruta   = plataformas en orden de ascenso (x, y, ancho)
#   extras = plataformas decorativas con fruta (no son parte de la ruta)
# ----------------------------------------------------------------
NIVELES = [
    dict(nombre="Selva Amazónica", animal="jaguar", escenario="selva",
         cielo=((200, 238, 235), (236, 250, 220)),
         plat=((176, 226, 160), (226, 202, 168)), fruta=(255, 190, 120),
         ruta=[(100, 525, 220), (390, 440, 200), (640, 355, 220),
               (430, 270, 170), (170, 185, 200), (440, 105, 240)],
         extras=[(880, 520, 120), (900, 300, 120)]),
    dict(nombre="Páramo Andino", animal="oso", escenario="paramo",
         cielo=((206, 224, 246), (246, 240, 232)),
         plat=((186, 220, 176), (214, 204, 184)), fruta=(255, 170, 190),
         ruta=[(820, 525, 200), (560, 440, 190), (300, 355, 190),
               (60, 270, 180), (300, 185, 200), (560, 100, 240)],
         extras=[(80, 500, 120), (850, 260, 150)]),
    dict(nombre="Costa Caribe", animal="perezoso", escenario="caribe",
         cielo=((178, 226, 250), (255, 244, 214)),
         plat=((255, 240, 190), (244, 214, 160)), fruta=(255, 205, 130),
         ruta=[(80, 530, 180), (330, 450, 160), (570, 370, 160),
               (810, 290, 170), (580, 205, 170), (340, 120, 200)],
         extras=[(880, 520, 150), (100, 300, 130)]),
    dict(nombre="Llanos Orientales", animal="chiguiro", escenario="llanos",
         cielo=((255, 224, 196), (255, 246, 214)),
         plat=((232, 232, 160), (222, 196, 150)), fruta=(255, 160, 150),
         ruta=[(900, 530, 150), (680, 450, 140), (460, 370, 140),
               (240, 290, 140), (40, 210, 140), (260, 130, 150)],
         extras=[(60, 520, 130), (700, 200, 130)]),
    dict(nombre="Nevados Andinos", animal="condor", escenario="nevado",
         cielo=((196, 214, 246), (240, 246, 255)),
         plat=((246, 250, 255), (150, 178, 224)), fruta=(255, 180, 210),
         ruta=[(100, 548, 150), (320, 473, 130), (530, 398, 130),
               (740, 323, 130), (940, 248, 120), (720, 173, 140),
               (480, 98, 160)],
         extras=[(900, 520, 150), (60, 300, 140)]),
]
MODOS = [
    ("Usuario vs PC", 1, 1),
    ("PC vs Usuario vs Usuario", 2, 1),
    ("Usuario vs Usuario", 2, 0),
]  # (nombre, humanos, pcs)
CARACT = ["dx", "dy", "en_suelo", "suelo_adelante"]
# ----------------------------------------------------------------
# CEREBRO DE LA PC (scikit-learn)
# ----------------------------------------------------------------
class CerebroPC:
    """Árbol de decisión que aprende a moverse en las plataformas.
    Se entrena con datos sintéticos (pandas) etiquetados por una regla
    "experta" con un poco de ruido, para que la PC cometa errores humanos.
    Acciones: 0 quieto, 1 izq, 2 der, 3 salto, 4 salto+izq, 5 salto+der
    """
    ACCIONES = {0: (0, False), 1: (-1, False), 2: (1, False),
                3: (0, True), 4: (-1, True), 5: (1, True)}
    def __init__(self, muestras=20000, semilla=7):
        self.modelo, self.precision = self._entrenar(muestras, semilla)
    @staticmethod
    def _experto(dx, dy, en_suelo, suelo_adelante):
        if abs(dx) <= 12:
            mover = 0
        else:
            mover = 1 if dx > 0 else -1
        saltar = bool(en_suelo) and (
            (not suelo_adelante and abs(dx) > 12) or (dy < -20 and abs(dx) < 60))
        if saltar:
            return {0: 3, -1: 4, 1: 5}[mover]
        return {0: 0, -1: 1, 1: 2}[mover]
    def _entrenar(self, n, semilla):
        rng = np.random.default_rng(semilla)
        df = pd.DataFrame({
            "dx": rng.uniform(-600, 600, n),
            "dy": rng.uniform(-300, 150, n),
            "en_suelo": rng.integers(0, 2, n),
            "suelo_adelante": rng.integers(0, 2, n),
        })
        df["accion"] = [self._experto(a, b, c, d) for a, b, c, d in
                        zip(df.dx, df.dy, df.en_suelo, df.suelo_adelante)]
        ruido = rng.random(n) < 0.01          # 1 % de errores "humanos"
        df.loc[ruido, "accion"] = rng.integers(0, 6, int(ruido.sum()))
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[CARACT].values, df["accion"].values, test_size=0.2, random_state=1)
        modelo = DecisionTreeClassifier(max_depth=12, min_samples_leaf=40, random_state=1).fit(X_tr, y_tr)
        return modelo, modelo.score(X_te, y_te)
    def decidir(self, dx, dy, en_suelo, suelo_adelante):
        a = int(self.modelo.predict(
            np.array([[dx, dy, int(en_suelo), int(suelo_adelante)]]))[0])
        return self.ACCIONES[a]
# ----------------------------------------------------------------
# OBJETOS DEL JUEGO
# ----------------------------------------------------------------
class Plataforma:
    def __init__(self, x, y, w, idx=None, h=18):
        self.rect = pygame.Rect(x, y, w, h)
        self.idx = idx  # índice en la ruta (None = decorativa)
class Jugador:
    ANCHO, ALTO = 34, 40
    def __init__(self, nombre, animal, teclas, es_pc, spawn_x, rol, vel):
        self.nombre = nombre
        self.animal = animal
        self.teclas = teclas
        self.es_pc = es_pc
        self.rol = rol          # "J1", "J2" o "PC" (para los colores)
        self.vel = vel
        self.spawn = (spawn_x, SUELO_Y - self.ALTO)
        self.puntos = 0
        self.reiniciar_nivel()
    def reiniciar_nivel(self):
        self.x, self.y = self.spawn
        self.vx = self.vy = 0.0
        self.en_suelo = False
        self.idx_plat = 0
        self.mira = 1
        self.fase = 0.0
        self.pts_nivel = 0
        self.frutas = 0
        self.llego = False
        self.objetivo = 1
        self.atasco = 0
        self.ultimo_x = self.x
        self.pausa = 0
        self.en_extra = False
    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.ANCHO, self.ALTO)
    @property
    def total(self):
        return self.puntos + self.pts_nivel
    def mover(self, plataformas, dirx, salto):
        self.vx = dirx * self.vel
        if dirx:
            self.mira = dirx
            self.fase += 0.3
        if salto and self.en_suelo:
            self.vy = FUERZA_SALTO
            self.en_suelo = False
        self.vy = min(self.vy + GRAVEDAD, 14)
        self.x = max(0, min(ANCHO - self.ANCHO, self.x + self.vx))
        base_previa = self.y + self.ALTO
        self.y += self.vy
        self.en_suelo = False
        if self.vy >= 0:  # plataformas "de una vía": solo se aterriza desde arriba
            mejor = None
            for p in plataformas:
                r = p.rect
                if r.right - 4 > self.x and r.left + 4 < self.x + self.ANCHO:
                    if base_previa <= r.top + 1 and self.y + self.ALTO >= r.top:
                        if mejor is None or r.top < mejor.rect.top:
                            mejor = p
            if mejor:
                self.y = mejor.rect.top - self.ALTO
                self.vy = 0
                self.en_suelo = True
                self.en_extra = mejor.idx is None
                if mejor.idx is not None:
                    self.idx_plat = mejor.idx
def construir_nivel(n):
    d = NIVELES[n]
    plats = [Plataforma(0, SUELO_Y, ANCHO, 0, h=ALTO - SUELO_Y)]
    for i, (x, y, w) in enumerate(d["ruta"], start=1):
        plats.append(Plataforma(x, y, w, i))
    extras = [Plataforma(x, y, w, None) for x, y, w in d["extras"]]
    ultima = plats[-1].rect
    bandera = pygame.Rect(ultima.centerx - 3, ultima.top - 60, 40, 60)
    frutas = [pygame.Rect(p.rect.centerx - 10, p.rect.top - 28, 20, 20)
              for p in plats[1:-1]]
    frutas += [pygame.Rect(p.rect.centerx - 10, p.rect.top - 28, 20, 20)
               for p in extras]
    frutas += [pygame.Rect(x, SUELO_Y - 28, 20, 20) for x in (300, 560, 800)]
    return plats, extras, bandera, frutas
def control_pc(j, plats, bandera, cerebro, cfg):
    """Calcula las características que ve la PC y deja que el árbol decida.
    cfg (dificultad) controla qué tanto duda la PC."""
    n = len(plats) - 1
    pies = j.y + j.ALTO
    if j.en_suelo:
        j.objetivo = j.idx_plat + 1
    if j.en_suelo and j.en_extra:      # cayó en una plataforma decorativa: volver al suelo
        tx, ttop = ANCHO / 2, SUELO_Y
    elif j.objetivo <= n:
        tx, ttop = plats[j.objetivo].rect.centerx, plats[j.objetivo].rect.top
    else:
        tx, ttop = bandera.centerx, plats[n].rect.top
    cx = j.x + j.ANCHO / 2
    dx, dy = tx - cx, ttop - pies
    d = 1 if dx > 0 else -1 if dx < 0 else j.mira
    sonda = pygame.Rect(int(cx + d * 40 - 2), int(pies + 2), 4, 12)
    suelo_adelante = any(p.rect.colliderect(sonda) for p in plats)
    dirx, salto = cerebro.decidir(dx, dy, j.en_suelo, suelo_adelante)
    # la PC a veces duda un momento (más en fácil, casi nada en difícil)
    if j.pausa > 0:
        j.pausa -= 1
        j.atasco = 0
        j.ultimo_x = j.x
        return 0, False
    if j.en_suelo and random.random() < cfg["p_pausa"]:
        j.pausa = random.randint(*cfg["pausa"])
    # anti-atasco: si intenta avanzar y no se mueve, salta
    if j.en_suelo and dirx != 0 and abs(j.x - j.ultimo_x) < 0.5:
        j.atasco += 1
    else:
        j.atasco = 0
    j.ultimo_x = j.x
    if j.atasco > 40:
        salto = True
    return dirx, salto
# ----------------------------------------------------------------
# DIBUJO: ANIMALES
# ----------------------------------------------------------------
def dibujar_animal(s, clave, rect, mira, fase=0.0):
    c = COLORES[clave]
    f = mira
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    cx = x + w // 2
    paso = math.sin(fase) * 4
    for k, off in enumerate((-8, 8)):  # patas
        dx = paso if k else -paso
        pygame.draw.rect(s, c["acento"], (cx + off - 4 + dx * 0.5, y + h - 11, 8, 11),
                         border_radius=4)
    # cola
    pygame.draw.circle(s, c["acento"], (cx - f * (w // 2 + 2), y + 24), 4)
    # cuerpo y cabeza
    pygame.draw.ellipse(s, c["cuerpo"], (x, y + 14, w, h - 20))
    hx, hy = cx + f * 8, y + 13
    pygame.draw.circle(s, c["cuerpo"], (hx, hy), 12)
    def ojo(ox=5, oy=-2, r=3):
        pygame.draw.circle(s, (255, 255, 255), (hx + f * ox, hy + oy), r)
        pygame.draw.circle(s, (70, 60, 70), (hx + f * (ox + 1), hy + oy), max(1, r - 1))
    if clave == "jaguar":
        for ox in (-8, 8):
            pygame.draw.circle(s, c["cuerpo"], (hx + ox, hy - 10), 5)
            pygame.draw.circle(s, (250, 190, 190), (hx + ox, hy - 10), 2)
        for ox, oy in ((6, 22), (16, 28), (25, 21), (11, 33), (27, 31)):
            pygame.draw.circle(s, c["acento"], (x + ox, y + oy), 3)
        pygame.draw.circle(s, c["acento"], (hx - 4, hy - 7), 2)
        ojo()
        pygame.draw.circle(s, (240, 160, 170), (hx + f * 11, hy + 3), 3)
    elif clave == "oso":
        for ox in (-9, 9):
            pygame.draw.circle(s, c["acento"], (hx + ox, hy - 10), 5)
        pygame.draw.ellipse(s, (250, 240, 226), (hx + f * 4 - 7, hy + 1, 14, 10))  # hocico
        pygame.draw.circle(s, (250, 240, 226), (hx + f * 5, hy - 2), 7, 2)           # anteojos
        pygame.draw.ellipse(s, (250, 240, 226), (cx - 8, y + 20, 16, 12))           # pecho
        ojo(5, -2, 2)
        pygame.draw.circle(s, (110, 90, 90), (hx + f * 11, hy + 4), 3)
    elif clave == "perezoso":
        pygame.draw.ellipse(s, c["acento"], (hx - 11, hy - 6, 22, 12))               # antifaz
        pygame.draw.line(s, c["acento"], (cx + f * 4, y + 20), (cx + f * 4, y + 34), 4)
        for k in (-2, 0, 2):
            pygame.draw.line(s, (120, 100, 90), (cx + f * 4 + k, y + 34), (cx + f * 4 + k, y + 38), 1)
        ojo(5, -1, 2)
        pygame.draw.arc(s, (110, 90, 90), (hx + f * 3 - 4, hy, 10, 8), 3.4, 6.0, 2)  # sonrisa
    elif clave == "chiguiro":
        pygame.draw.ellipse(s, c["cuerpo"], (hx + f * 4 - 9, hy - 2, 20, 15))        # hocico
        for ox in (-8, 8):
            pygame.draw.circle(s, c["acento"], (hx + ox, hy - 10), 3)
        ojo(3, -4, 2)
        pygame.draw.circle(s, (120, 95, 90), (hx + f * 12, hy + 3), 3)
    elif clave == "condor":
        pygame.draw.ellipse(s, c["acento"], (x + 3, y + 17, w - 10, 17))             # ala
        pygame.draw.ellipse(s, (255, 255, 255), (cx - 11 + f * 4, y + 17, 22, 8))    # collar
        pygame.draw.circle(s, (246, 214, 214), (hx, hy), 10)                          # cabeza
        pygame.draw.polygon(s, (255, 226, 130), [(hx + f * 8, hy - 3), (hx + f * 18, hy + 3),
                                                   (hx + f * 8, hy + 6)])
        ojo(3, -2, 2)
    elif clave == "rana":
        for ox in (-6, 6):
            pygame.draw.circle(s, c["cuerpo"], (hx + ox, hy - 10), 6)
            pygame.draw.circle(s, (255, 255, 255), (hx + ox, hy - 10), 4)
            pygame.draw.circle(s, (70, 60, 70), (hx + ox + f, hy - 10), 2)
        for ox, oy in ((6, 24), (15, 29), (24, 23)):
            pygame.draw.circle(s, c["acento"], (x + ox, y + oy), 3)
        pygame.draw.arc(s, (150, 110, 100), (hx - 8, hy - 1, 16, 10), 3.4, 6.0, 2)
# ----------------------------------------------------------------
# DIBUJO: ESCENARIOS
# ----------------------------------------------------------------
def degradado(w, h, c1, c2):
    s = pygame.Surface((w, h))
    for y in range(h):
        t = y / h
        pygame.draw.line(s, [int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)], (0, y), (w, y))
    return s
def nube(s, x, y, k=1.0):
    for dx, dy, r in ((0, 0, 22), (24, -8, 26), (52, 0, 22), (26, 8, 20)):
        pygame.draw.circle(s, (255, 255, 255), (int(x + dx * k), int(y + dy * k)), int(r * k))
def palmera(s, x, base, h, tronco, hoja):
    top = (x + h // 6, base - h)
    pygame.draw.line(s, tronco, (x, base), top, 9)
    for ang in (-160, -125, -90, -55, -20):
        a = math.radians(ang)
        fin = (top[0] + math.cos(a) * 55, top[1] + math.sin(a) * 30 + 18)
        pygame.draw.line(s, hoja, top, fin, 7)
def crear_fondo(n, con_plataformas=True):
    d = NIVELES[n]
    rng = random.Random(100 + n)
    s = degradado(ANCHO, ALTO, *d["cielo"])
    for _ in range(5):
        nube(s, rng.randint(0, ANCHO), rng.randint(40, 220), rng.uniform(0.7, 1.3))
    e = d["escenario"]
    if e == "selva":
        pygame.draw.circle(s, (255, 248, 200), (920, 100), 50)
        for x0 in range(-120, ANCHO, 270):
            pygame.draw.ellipse(s, (196, 232, 196), (x0, 400, 400, 320))
        for x in range(10, ANCHO, 95):
            h = rng.randint(130, 210)
            pygame.draw.rect(s, (224, 204, 172), (x + 22, SUELO_Y - h, 12, h))
            for ox, oy, r, col in ((28, 0, 40, (170, 222, 170)), (6, 14, 28, (184, 230, 176)),
                                   (50, 14, 28, (160, 216, 168))):
                pygame.draw.circle(s, col, (x + ox, SUELO_Y - h + oy), r)
    elif e == "paramo":
        for cx_, top, w_ in ((150, 250, 300), (480, 200, 360), (850, 260, 320)):
            pygame.draw.polygon(s, (204, 218, 238), [(cx_ - w_, SUELO_Y), (cx_, top), (cx_ + w_, SUELO_Y)])
            pygame.draw.polygon(s, (255, 255, 255), [(cx_ - 60, top + 62), (cx_, top), (cx_ + 60, top + 62)])
        for x in range(30, ANCHO, 85):
            h = rng.randint(40, 90)
            pygame.draw.rect(s, (216, 202, 160), (x, SUELO_Y - h, 9, h))
            for ang in range(0, 360, 40):
                a = math.radians(ang)
                pygame.draw.line(s, (176, 214, 170), (x + 4, SUELO_Y - h),
                                 (x + 4 + math.cos(a) * 20, SUELO_Y - h + math.sin(a) * 12), 4)
            pygame.draw.circle(s, (204, 228, 184), (x + 4, SUELO_Y - h), 6)
    elif e == "caribe":
        pygame.draw.circle(s, (255, 238, 190), (190, 310), 55)
        pygame.draw.rect(s, (170, 224, 238), (0, 360, ANCHO, 140))
        for yy in range(372, 496, 18):
            for xx in range(rng.randint(0, 50), ANCHO, 120):
                pygame.draw.line(s, (206, 242, 248), (xx, yy), (xx + 40, yy), 3)
        for x in (60, 330, 620, 900, 1040):
            palmera(s, x, SUELO_Y, rng.randint(150, 220), (226, 200, 160), (156, 216, 172))
    elif e == "llanos":
        pygame.draw.circle(s, (255, 222, 168), (550, 450), 90)
        pygame.draw.rect(s, (246, 236, 178), (0, 450, ANCHO, SUELO_Y - 450))
        for x0 in range(-80, ANCHO, 240):
            pygame.draw.ellipse(s, (238, 228, 160), (x0, 500, 360, 220))
        for x in (120, 420, 760, 1000):
            palmera(s, x, SUELO_Y, rng.randint(140, 190), (214, 190, 150), (180, 218, 160))
        for _ in range(6):
            bx, by = rng.randint(50, ANCHO - 50), rng.randint(60, 200)
            pygame.draw.arc(s, (190, 170, 160), (bx, by, 16, 10), 0.3, 2.8, 2)
            pygame.draw.arc(s, (190, 170, 160), (bx + 14, by, 16, 10), 0.3, 2.8, 2)
    elif e == "nevado":
        for cx_, top, w_, col in ((120, 180, 340, (206, 220, 244)), (900, 150, 380, (206, 220, 244)),
                                  (500, 90, 420, (196, 212, 240))):
            pygame.draw.polygon(s, col, [(cx_ - w_, SUELO_Y), (cx_, top), (cx_ + w_, SUELO_Y)])
            pygame.draw.polygon(s, (255, 255, 255), [(cx_ - 90, top + 100), (cx_, top), (cx_ + 90, top + 100)])
        for _ in range(60):
            pygame.draw.circle(s, (255, 255, 255), (rng.randint(0, ANCHO), rng.randint(0, SUELO_Y)), rng.randint(1, 3))
    # suelo
    top_c, body_c = d["plat"]
    pygame.draw.rect(s, body_c, (0, SUELO_Y, ANCHO, ALTO - SUELO_Y))
    pygame.draw.rect(s, top_c, (0, SUELO_Y, ANCHO, 10))
    # plataformas (estáticas)
    borde = tuple(int(c * 0.74) for c in body_c)
    pygame.draw.line(s, borde, (0, SUELO_Y), (ANCHO, SUELO_Y), 2)
    for (x, y, w) in (d["ruta"] + d["extras"] if con_plataformas else []):
        pygame.draw.rect(s, body_c, (x, y, w, 18), border_radius=8)
        pygame.draw.rect(s, top_c, (x, y, w, 8), border_radius=8)
        pygame.draw.rect(s, borde, (x, y, w, 18), 2, border_radius=8)
    return s
def dibujar_fruta(s, r, color):
    pygame.draw.circle(s, color, r.center, 10)
    pygame.draw.circle(s, (255, 255, 255), (r.centerx - 3, r.centery - 3), 3)
    pygame.draw.ellipse(s, (150, 215, 150), (r.centerx, r.top - 5, 9, 5))
def dibujar_bandera(s, b, t):
    pygame.draw.line(s, (200, 190, 180), (b.x + 3, b.top), (b.x + 3, b.bottom), 4)
    pygame.draw.circle(s, (255, 236, 150), (b.x + 3, b.top), 4)
    onda = math.sin(t * 0.1) * 2
    pygame.draw.rect(s, (255, 236, 150), (b.x + 5, b.top + 2, 34, 11))
    pygame.draw.rect(s, (160, 190, 238), (b.x + 5, b.top + 13 + onda, 34, 6))
    pygame.draw.rect(s, (242, 160, 164), (b.x + 5, b.top + 19 + onda, 34, 6))
# ----------------------------------------------------------------
# RANKING (pandas + CSV): partidas ganadas y perdidas por jugador
# ----------------------------------------------------------------
def cargar_ranking(maximo=8):
    """Devuelve un DataFrame: nombre, ganadas, perdidas (ordenado)."""
    try:
        if not os.path.exists(ARCHIVO_RANKING):
            return pd.DataFrame(columns=["nombre", "ganadas", "perdidas"])
        df = pd.read_csv(ARCHIVO_RANKING)
        g = df.groupby("nombre")["resultado"].agg(
            ganadas=lambda s: int((s == "Ganada").sum()),
            perdidas=lambda s: int((s == "Perdida").sum())).reset_index()
        g = g.sort_values(["ganadas", "perdidas"], ascending=[False, True])
        return g.head(maximo).reset_index(drop=True)
    except Exception as ex:
        print("No se pudo leer el ranking:", ex)
        return pd.DataFrame(columns=["nombre", "ganadas", "perdidas"])
# ----------------------------------------------------------------
# JUEGO
# ----------------------------------------------------------------
class Juego:
    def __init__(self):
        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Aventura Salvaje Colombia")
        self.reloj = pygame.time.Clock()
        nombres = ["arial", "dejavusans", "helvetica"]
        self.f_gde = pygame.font.SysFont(nombres, 58, bold=True)
        self.f_med = pygame.font.SysFont(nombres, 30, bold=True)
        self.f_ayuda = pygame.font.SysFont(nombres, 22, bold=True)
        self.f_peq = pygame.font.SysFont(nombres, 20)
        self.f_min = pygame.font.SysFont(nombres, 16, bold=True)
        self.cerebro = CerebroPC()
        self.estado = "menu"
        self.modo = 0
        self.dif = 1
        self.cfg = DIFICULTADES[1]
        self.nombres = []
        self.entrada = ""
        self.fondo_menu = crear_fondo(0, con_plataformas=False)
        self.jugadores = []
        self.t = 0
        self.particulas = []
        self.supero = False
    # ---------- utilidades ----------
    def texto(self, txt, fuente, color, pos, centro=False, sombra=True):
        if sombra:
            sh_col = (40, 40, 50) if sum(color) > 600 else (255, 255, 255)
            sh = fuente.render(txt, True, sh_col)
            r = sh.get_rect(center=(pos[0] + 2, pos[1] + 2)) if centro else sh.get_rect(topleft=(pos[0] + 2, pos[1] + 2))
            self.pantalla.blit(sh, r)
        img = fuente.render(txt, True, color)
        r = img.get_rect(center=pos) if centro else img.get_rect(topleft=pos)
        self.pantalla.blit(img, r)
    def panel(self, rect, color=(255, 255, 255), alpha=170):
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=16)
        self.pantalla.blit(s, (rect[0], rect[1]))
    def ayuda(self, txt, y, color=(255, 255, 255), fondo=(45, 65, 90)):
        """Barra oscura con texto claro y en negrita: se lee sobre cualquier fondo."""
        img = self.f_ayuda.render(txt, True, color)
        r = img.get_rect(center=(ANCHO // 2, y))
        caja = r.inflate(34, 14)
        pygame.draw.rect(self.pantalla, fondo, caja, border_radius=14)
        pygame.draw.rect(self.pantalla, (255, 255, 255), caja, 2, border_radius=14)
        self.pantalla.blit(img, r)
    def boton(self, rect, txt, sel):
        """Botón de alto contraste: naranja intenso si está seleccionado, azul oscuro si no."""
        r = pygame.Rect(rect)
        relleno = (255, 130, 40) if sel else (40, 100, 130)
        borde = (120, 50, 0) if sel else (15, 45, 65)
        pygame.draw.rect(self.pantalla, borde, r.move(0, 4), border_radius=20)  # sombra
        pygame.draw.rect(self.pantalla, relleno, r, border_radius=20)
        pygame.draw.rect(self.pantalla, (255, 255, 255), r, 4 if sel else 2, border_radius=20)
        if sel:  # flecha indicadora
            cy = r.centery
            pygame.draw.polygon(self.pantalla, (255, 255, 255),
                                [(r.x + 16, cy - 10), (r.x + 16, cy + 10), (r.x + 32, cy)])
        self.texto(txt, self.f_med, (255, 255, 255), r.center, True, False)
    def chispas(self, x, y, color, n=12):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            v = random.uniform(1, 4)
            self.particulas.append([x, y, math.cos(a) * v, math.sin(a) * v - 1, 30, color])
    # ---------- bucle principal ----------
    def ejecutar(self):
        while True:
            self.t += 1
            eventos = pygame.event.get()
            for e in eventos:
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if self.estado == "menu":
                self.menu(eventos)
            elif self.estado == "dificultad":
                self.pantalla_dificultad(eventos)
            elif self.estado == "nombres":
                self.pantalla_nombres(eventos)
            elif self.estado == "seleccion":
                self.seleccion(eventos)
            elif self.estado == "jugando":
                self.jugar(eventos)
            elif self.estado == "resumen":
                self.resumen(eventos)
            elif self.estado == "final":
                self.final(eventos)
            elif self.estado == "ranking":
                self.pantalla_ranking(eventos)
            pygame.display.flip()
            self.reloj.tick(FPS)
    # ---------- MENÚ ----------
    def elegir_modo(self):
        _, humanos, pcs = MODOS[self.modo]
        if pcs:
            self.estado = "dificultad"
        else:
            self.cfg = DIFICULTADES[1]
            self.iniciar_nombres()
    def menu(self, eventos):
        for e in eventos:
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_UP, pygame.K_w):
                    self.modo = (self.modo - 1) % 3
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    self.modo = (self.modo + 1) % 3
                elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.modo = e.key - pygame.K_1
                    self.elegir_modo()
                elif e.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.elegir_modo()
                elif e.key == pygame.K_r:
                    self.estado = "ranking"
        self.pantalla.blit(self.fondo_menu, (0, 0))
        self.panel((150, 40, 800, 570), alpha=150)
        self.texto("AVENTURA SALVAJE", self.f_gde, (60, 120, 90), (ANCHO // 2, 100), True)
        self.texto("COLOMBIA", self.f_gde, (225, 120, 50), (ANCHO // 2, 160), True)
        self.ayuda("Elige el modo de juego:  ↑ ↓ y ENTER  ·  o teclas 1 - 2 - 3", 218)
        for i, (nombre, _, _) in enumerate(MODOS):
            self.boton((260, 250 + i * 80, 580, 62), f"{i + 1}.  {nombre}", i == self.modo)
        for k, clave in enumerate(ANIMALES.index):
            rect = pygame.Rect(210 + k * 125, 500, 34, 40)
            dibujar_animal(self.pantalla, clave, rect, 1, self.t * 0.15 + k)
        self.ayuda("5 niveles · Frutas +1 · Bandera +5   |   R: Ver ranking", 580)
    # ---------- DIFICULTAD ----------
    def pantalla_dificultad(self, eventos):
        for e in eventos:
            if e.type != pygame.KEYDOWN:
                continue
            if e.key == pygame.K_ESCAPE:
                self.estado = "menu"
                return
            if e.key in (pygame.K_UP, pygame.K_w):
                self.dif = (self.dif - 1) % 3
            elif e.key in (pygame.K_DOWN, pygame.K_s):
                self.dif = (self.dif + 1) % 3
            elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.dif = e.key - pygame.K_1
                self.cfg = DIFICULTADES[self.dif]
                self.iniciar_nombres()
                return
            elif e.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.cfg = DIFICULTADES[self.dif]
                self.iniciar_nombres()
                return
        self.pantalla.blit(self.fondo_menu, (0, 0))
        self.panel((150, 40, 800, 570), alpha=150)
        self.texto("DIFICULTAD", self.f_gde, (60, 120, 90), (ANCHO // 2, 105), True)
        self.ayuda("Elige la intensidad:  ↑ ↓ y ENTER  ·  o teclas 1 - 2 - 3", 175)
        for i, d in enumerate(DIFICULTADES):
            self.boton((260, 230 + i * 90, 580, 66), f"{i + 1}.  {d['nombre']}  ·  {d['desc']}", i == self.dif)
        self.ayuda("ESC: volver al menú", 560)
    # ---------- NOMBRES ----------
    def iniciar_nombres(self):
        self.nombres = []
        self.entrada = ""
        self.estado = "nombres"
    def pantalla_nombres(self, eventos):
        humanos = MODOS[self.modo][1]
        for e in eventos:
            if e.type != pygame.KEYDOWN:
                continue
            if e.key == pygame.K_ESCAPE:
                self.estado = "menu"
                return
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                nom = self.entrada.strip() or f"Jugador {len(self.nombres) + 1}"
                if nom.lower() in [n.lower() for n in self.nombres]:
                    nom = nom[:MAX_NOMBRE - 2] + " 2"
                self.nombres.append(nom)
                self.entrada = ""
                if len(self.nombres) == humanos:
                    self.iniciar_seleccion()
                    return
            elif e.key == pygame.K_BACKSPACE:
                self.entrada = self.entrada[:-1]
            elif e.unicode and e.unicode.isprintable() and len(self.entrada) < MAX_NOMBRE:
                self.entrada += e.unicode
        self.pantalla.blit(self.fondo_menu, (0, 0))
        self.panel((150, 40, 800, 570), alpha=150)
        self.texto("¿CÓMO TE LLAMAS?", self.f_gde, (60, 120, 90), (ANCHO // 2, 110), True)
        n = len(self.nombres) + 1
        self.texto(f"Jugador {n}: escribe tu nombre", self.f_med, (225, 120, 50), (ANCHO // 2, 220), True)
        caja = pygame.Rect(300, 270, 500, 70)
        pygame.draw.rect(self.pantalla, (255, 255, 255), caja, border_radius=16)
        pygame.draw.rect(self.pantalla, (40, 100, 130), caja, 4, border_radius=16)
        cursor = "|" if (self.t // 30) % 2 == 0 else ""
        self.texto(self.entrada + cursor, self.f_gde, (40, 60, 80), caja.center, True, False)
        self.ayuda(f"Escribe tu nombre (máx. {MAX_NOMBRE} letras) y presiona ENTER", 400)
        if self.nombres:
            self.ayuda("Jugador 1: " + self.nombres[0], 460, fondo=(225, 120, 50))
        self.ayuda("ESC: volver al menú", 560)
    # ---------- SELECCIÓN DE ANIMAL ----------
    def iniciar_seleccion(self):
        self.estado = "seleccion"
        humanos = MODOS[self.modo][1]
        self.cursor = [0, 1][:humanos]
        self.listo = [False] * humanos
    def seleccion(self, eventos):
        humanos = MODOS[self.modo][1]
        claves = list(ANIMALES.index)
        for e in eventos:
            if e.type != pygame.KEYDOWN:
                continue
            if e.key == pygame.K_ESCAPE:
                self.estado = "menu"
                return
            mapa = []
            if humanos == 1:
                mapa.append((0, (pygame.K_a, pygame.K_LEFT), (pygame.K_d, pygame.K_RIGHT),
                             (pygame.K_w, pygame.K_UP, pygame.K_SPACE)))
            else:
                mapa.append((0, (pygame.K_a,), (pygame.K_d,), (pygame.K_w,)))
                mapa.append((1, (pygame.K_LEFT,), (pygame.K_RIGHT,), (pygame.K_UP,)))
            for i, izq, der, ok in mapa:
                if self.listo[i]:
                    continue
                if e.key in izq:
                    self.cursor[i] = (self.cursor[i] - 1) % len(claves)
                elif e.key in der:
                    self.cursor[i] = (self.cursor[i] + 1) % len(claves)
                elif e.key in ok:
                    otros = [self.cursor[k] for k in range(humanos) if k != i and self.listo[k]]
                    if self.cursor[i] not in otros:
                        self.listo[i] = True
            if e.key == pygame.K_RETURN and all(self.listo):
                self.crear_partida()
                return
        self.pantalla.blit(self.fondo_menu, (0, 0))
        self.panel((60, 40, 980, 570), alpha=160)
        self.texto("Elige tu animal", self.f_gde, (60, 120, 90), (ANCHO // 2, 90), True)
        cols = [(225, 120, 50), (60, 110, 210)]
        for i in range(humanos):
            cl = claves[self.cursor[i]]
            x0 = 120 + i * 480 if humanos == 2 else 330
            self.panel((x0, 150, 420, 330), color=(255, 250, 240), alpha=230)
            self.texto(self.nombres[i] + ("  · ¡LISTO!" if self.listo[i] else ""), self.f_med, cols[i],
                       (x0 + 210, 180), True)
            dibujar_animal(self.pantalla, cl, pygame.Rect(x0 + 175, 220, 68, 80), 1, self.t * 0.1)
            fila = ANIMALES.loc[cl]
            self.texto(fila["nombre"], self.f_med, (90, 90, 110), (x0 + 210, 335), True, False)
            self.texto(fila["region"], self.f_peq, (70, 130, 90), (x0 + 210, 365), True, False)
            palabras, lineas, ln = fila["dato"].split(), [], ""
            for p in palabras:
                if len(ln + p) > 40:
                    lineas.append(ln)
                    ln = ""
                ln += p + " "
            lineas.append(ln)
            for k, l in enumerate(lineas[:3]):
                self.texto(l.strip(), self.f_min, (100, 100, 120), (x0 + 210, 395 + k * 20), True, False)
        if humanos == 2:
            self.ayuda(f"{self.nombres[0]}:  A / D elegir  ·  W confirmar", 507, fondo=(225, 120, 50))
            self.ayuda(f"{self.nombres[1]}:  ← / → elegir  ·  ↑ confirmar", 541, fondo=(60, 110, 210))
        else:
            self.ayuda("← / → o A / D: elegir   ·   ↑ / W / ESPACIO: confirmar", 520)
        if all(self.listo):
            self.ayuda("Presiona ENTER para comenzar", 583, fondo=(225, 120, 50))
        else:
            self.ayuda("ESC: volver", 583)
    # ---------- PARTIDA ----------
    def crear_partida(self):
        _, humanos, pcs = MODOS[self.modo]
        claves = list(ANIMALES.index)
        elegidos = [claves[c] for c in self.cursor]
        libres = [c for c in claves if c not in elegidos]
        random.shuffle(libres)
        self.jugadores = []
        if humanos == 1:
            t1 = dict(izq=(pygame.K_a, pygame.K_LEFT), der=(pygame.K_d, pygame.K_RIGHT),
                      salto=(pygame.K_w, pygame.K_UP, pygame.K_SPACE))
            self.jugadores.append(Jugador(self.nombres[0], elegidos[0], t1, False, 30, "J1", VEL_JUGADOR))
        else:
            t1 = dict(izq=(pygame.K_a,), der=(pygame.K_d,), salto=(pygame.K_w,))
            t2 = dict(izq=(pygame.K_LEFT,), der=(pygame.K_RIGHT,), salto=(pygame.K_UP,))
            self.jugadores.append(Jugador(self.nombres[0], elegidos[0], t1, False, 30, "J1", VEL_JUGADOR))
            self.jugadores.append(Jugador(self.nombres[1], elegidos[1], t2, False, 90, "J2", VEL_JUGADOR))
        if pcs:
            self.jugadores.append(Jugador("PC", libres[0], None, True, 150 if humanos == 2 else 90,
                                          "PC", self.cfg["vel"]))
        self.iniciar_nivel(0)
    def iniciar_nivel(self, n):
        self.nivel = n
        self.plats, self.extras, self.bandera, self.frutas = construir_nivel(n)
        self.todas = self.plats + self.extras
        self.fondo = crear_fondo(n)
        for j in self.jugadores:
            j.reiniciar_nivel()
        self.tiempo = self.cfg["tiempo"] * FPS
        self.ganador = None
        self.cuenta_final = 0
        self.particulas = []
        self.estado = "jugando"
    def jugar(self, eventos):
        for e in eventos:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.estado = "menu"
                return
        teclas = pygame.key.get_pressed()
        congelado = self.ganador is not None
        for j in self.jugadores:
            if congelado:
                dirx, salto = 0, False
            elif j.es_pc:
                dirx, salto = control_pc(j, self.plats, self.bandera, self.cerebro, self.cfg)
            else:
                dirx = int(any(teclas[k] for k in j.teclas["der"])) - int(any(teclas[k] for k in j.teclas["izq"]))
                salto = any(teclas[k] for k in j.teclas["salto"])
            j.mover(self.todas, dirx, salto)
            r = j.rect
            for fr in self.frutas[:]:
                if r.colliderect(fr):
                    self.frutas.remove(fr)
                    j.frutas += 1
                    j.pts_nivel += PUNTOS_FRUTA
                    self.chispas(fr.centerx, fr.centery, NIVELES[self.nivel]["fruta"])
            if not congelado and r.colliderect(self.bandera):
                j.llego = True
                j.pts_nivel += PUNTOS_BANDERA
                self.ganador = j
                self.cuenta_final = 110
                for _ in range(5):
                    self.chispas(self.bandera.centerx, self.bandera.top, random.choice(
                        [(255, 236, 150), (160, 190, 238), (242, 160, 164)]), 14)
        if self.ganador is None:
            self.tiempo -= 1
            if self.tiempo <= 0:
                self.cuenta_final = 1
                self.ganador = False  # tiempo agotado
        if self.ganador is not None:
            self.cuenta_final -= 1
            if self.cuenta_final <= 0:
                self.terminar_nivel()
        self.dibujar_juego()
    def dibujar_juego(self):
        s = self.pantalla
        d = NIVELES[self.nivel]
        s.blit(self.fondo, (0, 0))
        dibujar_bandera(s, self.bandera, self.t)
        for fr in self.frutas:
            dibujar_fruta(s, fr, d["fruta"])
        cols = {"J1": (225, 120, 50), "J2": (60, 110, 210), "PC": (70, 150, 80)}
        for j in self.jugadores:
            dibujar_animal(s, j.animal, j.rect, j.mira, j.fase)
            etq = self.f_min.render(j.nombre[:9], True, (255, 255, 255))
            bg = pygame.Rect(0, 0, etq.get_width() + 12, 20)
            bg.midbottom = (j.rect.centerx, j.rect.top - 8)
            pygame.draw.rect(s, cols[j.rol], bg, border_radius=10)
            s.blit(etq, etq.get_rect(center=bg.center))
        for p in self.particulas[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.12
            p[4] -= 1
            if p[4] <= 0:
                self.particulas.remove(p)
            else:
                pygame.draw.circle(s, p[5], (int(p[0]), int(p[1])), 3)
        # HUD
        for i, j in enumerate(self.jugadores):
            self.panel((10 + i * 205, 6, 198, 30), alpha=210)
            self.texto(f"{j.nombre[:9]}: {j.total} pts ({j.frutas} fr.)", self.f_min,
                       cols[j.rol], (20 + i * 205, 13), False, False)
        self.panel((640, 6, 300, 30), alpha=210)
        self.texto(f"Nivel {self.nivel + 1}/5 · {d['nombre']}", self.f_min, (80, 80, 100),
                   (790, 21), True, False)
        self.panel((950, 6, 140, 30), alpha=210)
        self.texto(f"Tiempo {max(0, self.tiempo // FPS)} s", self.f_min, (80, 80, 100),
                   (1020, 21), True, False)
        if self.ganador:
            self.texto(f"¡{self.ganador.nombre} llegó a la meta!", self.f_med, (225, 120, 50),
                       (ANCHO // 2, 80), True)
        elif self.ganador is False:
            self.texto("¡Se acabó el tiempo!", self.f_med, (225, 120, 50), (ANCHO // 2, 80), True)
    # ---------- RESUMEN DE NIVEL ----------
    def registrar_resultados(self):
        """Guarda en el CSV una fila por jugador humano: partida Ganada o Perdida."""
        filas = [dict(fecha=datetime.now().strftime("%Y-%m-%d %H:%M"), nombre=j.nombre,
                      modo=MODOS[self.modo][0],
                      dificultad=self.cfg["nombre"] if MODOS[self.modo][2] else "-",
                      nivel=self.nivel + 1,
                      resultado="Ganada" if self.resumen_ganador is j else "Perdida")
                 for j in self.jugadores if not j.es_pc]
        nuevo = pd.DataFrame(filas)
        try:
            if os.path.exists(ARCHIVO_RANKING):
                df = pd.concat([pd.read_csv(ARCHIVO_RANKING), nuevo], ignore_index=True)
            else:
                df = nuevo
            df.to_csv(ARCHIVO_RANKING, index=False)
        except Exception as ex:  # el juego no debe fallar por el CSV
            print("No se pudo guardar el ranking:", ex)
    def terminar_nivel(self):
        g = self.ganador
        self.resumen_ganador = g if g else None
        # Solo se pasa de nivel si un humano llega primero (le ganó a la PC)
        self.supero = bool(g) and not g.es_pc
        self.pts_ganados = {j.nombre: (j.pts_nivel if self.supero else 0) for j in self.jugadores}
        for j in self.jugadores:
            if self.supero:
                j.puntos += j.pts_nivel
            j.pts_nivel = 0
        self.registrar_resultados()
        self.estado = "resumen"
    def resumen(self, eventos):
        for e in eventos:
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_SPACE):
                if not self.supero:
                    self.iniciar_nivel(self.nivel)      # repetir el mismo nivel
                elif self.nivel + 1 < len(NIVELES):
                    self.iniciar_nivel(self.nivel + 1)
                else:
                    self.estado = "final"
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.estado = "menu"
                return
        d = NIVELES[self.nivel]
        self.pantalla.blit(self.fondo, (0, 0))
        self.panel((180, 50, 740, 550), alpha=235)
        if self.supero:
            self.texto(f"Nivel {self.nivel + 1} completado", self.f_gde, (60, 120, 90), (ANCHO // 2, 105), True)
        else:
            self.texto("¡Nivel no superado!", self.f_gde, (200, 70, 60), (ANCHO // 2, 105), True)
        g = self.resumen_ganador
        self.texto(f"Llegó primero: {g.nombre}" if g else "Nadie llegó a la meta", self.f_med,
                   (225, 120, 50), (ANCHO // 2, 160), True)
        if not self.supero:
            hay_pc = any(j.es_pc for j in self.jugadores)
            msg = "Debes ganarle a la PC para pasar de nivel" if hay_pc and g else "Se acabó el tiempo: repite el nivel"
            self.ayuda(msg, 200, fondo=(200, 70, 60))
        for i, j in enumerate(sorted(self.jugadores, key=lambda x: -x.puntos)):
            y = 240 + i * 55
            dibujar_animal(self.pantalla, j.animal, pygame.Rect(240, y, 34, 40), 1, self.t * 0.1)
            self.texto(f"{j.nombre}  (+{self.pts_ganados[j.nombre]} en este nivel)   Total: {j.puntos}",
                       self.f_ayuda, (70, 70, 90), (300, y + 10), False, False)
        a = ANIMALES.loc[d["animal"]]
        self.texto(f"Dato curioso · {a['nombre']}", self.f_med, (70, 130, 90), (ANCHO // 2, 440), True, False)
        ln, lineas = "", []
        for p in a["dato"].split():
            if len(ln + p) > 62:
                lineas.append(ln)
                ln = ""
            ln += p + " "
        lineas.append(ln)
        for k, l in enumerate(lineas):
            self.texto(l.strip(), self.f_peq, (90, 90, 110), (ANCHO // 2, 480 + k * 26), True, False)
        self.ayuda("ENTER: continuar" if self.supero else "ENTER: reintentar el nivel", 568,
                   fondo=(225, 120, 50))
    # ---------- TABLA DE RANKING ----------
    def tabla_ranking(self, y0, paso, fuente):
        df = cargar_ranking(8 if self.estado == "ranking" else 5)
        self.texto("#", self.f_min, (60, 110, 140), (300, y0), False, False)
        self.texto("JUGADOR", self.f_min, (60, 110, 140), (350, y0), False, False)
        self.texto("GANADAS", self.f_min, (60, 110, 140), (700, y0), True, False)
        self.texto("PERDIDAS", self.f_min, (60, 110, 140), (850, y0), True, False)
        if df.empty:
            self.texto("Aún no hay partidas registradas", fuente, (90, 90, 110),
                       (ANCHO // 2, y0 + paso + 10), True, False)
            return
        for i, fila in df.iterrows():
            y = y0 + paso * (i + 1)
            self.texto(str(i + 1), fuente, (70, 70, 90), (300, y), False, False)
            self.texto(str(fila["nombre"]), fuente, (70, 70, 90), (350, y), False, False)
            self.texto(str(fila["ganadas"]), fuente, (50, 140, 70), (700, y), True, False)
            self.texto(str(fila["perdidas"]), fuente, (200, 70, 60), (850, y), True, False)
    def pantalla_ranking(self, eventos):
        for e in eventos:
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_SPACE):
                self.estado = "menu"
                return
        self.pantalla.blit(self.fondo_menu, (0, 0))
        self.panel((150, 40, 800, 570), alpha=235)
        self.texto("RANKING", self.f_gde, (60, 120, 90), (ANCHO // 2, 100), True)
        self.tabla_ranking(170, 46, self.f_med)
        self.ayuda("ENTER o ESC: volver al menú", 580)
    # ---------- FINAL ----------
    def final(self, eventos):
        for e in eventos:
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_SPACE):
                self.estado = "menu"
                return
        self.pantalla.blit(self.fondo_menu, (0, 0))
        self.panel((120, 30, 860, 590), alpha=235)
        orden = sorted(self.jugadores, key=lambda x: -x.puntos)
        empate = len(orden) > 1 and orden[0].puntos == orden[1].puntos
        self.texto("¡FIN DEL JUEGO!", self.f_gde, (60, 120, 90), (ANCHO // 2, 85), True)
        self.texto("¡Empate!" if empate else f"¡Ganó {orden[0].nombre}!", self.f_med, (225, 120, 50),
                   (ANCHO // 2, 140), True)
        for i, j in enumerate(orden):
            y = 180 + i * 55
            dibujar_animal(self.pantalla, j.animal, pygame.Rect(260, y, 34, 40), 1, self.t * 0.1)
            self.texto(f"{i + 1}. {j.nombre} - {ANIMALES.loc[j.animal, 'nombre']}: {j.puntos} puntos",
                       self.f_med, (70, 70, 90), (320, y + 6), False, False)
        self.texto("Ranking: partidas ganadas y perdidas", self.f_med, (70, 130, 90),
                   (ANCHO // 2, 375), True, False)
        self.tabla_ranking(405, 27, self.f_peq)
        self.ayuda("ENTER: volver al menú", 592, fondo=(225, 120, 50))
if __name__ == "__main__":
    Juego().ejecutar()
