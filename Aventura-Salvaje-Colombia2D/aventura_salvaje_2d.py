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
# SONIDO: todo se sintetiza con numpy (no hace falta ningún archivo)
# ----------------------------------------------------------------
SR = 44100
VOL_MUSICA = 0.30
VOL_EFECTOS = 0.55
def _frec(m):
    return 440.0 * 2 ** ((m - 69) / 12)
def _osc(ph, tipo):
    if tipo == "tri":
        return (2 / np.pi) * np.arcsin(np.sin(ph))
    if tipo == "cuad":
        return 0.6 * np.sign(np.sin(ph))
    return np.sin(ph)
def _nota(f, dur, tipo="sin", vol=0.5, decay=4.0, ataque=0.008, f_fin=None):
    n = int(SR * dur)
    t = np.arange(n) / SR
    fr = np.full(n, float(f)) if f_fin is None else np.linspace(f, f_fin, n)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    env = np.minimum(1.0, t / ataque) * np.exp(-decay * t)
    cola = min(n, int(SR * 0.01))
    env[-cola:] *= np.linspace(1, 0, cola)
    return _osc(ph, tipo) * env * vol
def _ruido(dur, vol, decay, suave=0):
    n = int(SR * dur)
    r = np.random.uniform(-1, 1, n)
    if suave:       # pasa-bajos: sonido grave y sordo
        r = np.convolve(r, np.ones(suave) / suave, "same")
    else:           # pasa-altos: sonido agudo (platillo)
        r = np.diff(r, prepend=0)
    return r * np.exp(-decay * np.arange(n) / SR) * vol
def _mezclar(partes, dur):
    buf = np.zeros(int(SR * dur))
    for ini, arr in partes:
        i = int(SR * ini)
        fin = min(len(buf), i + len(arr))
        if fin > i:
            buf[i:fin] += arr[:fin - i]
    return buf
def _efectos():
    return {
        "salto":    _nota(320, 0.16, "tri", 0.45, 6, f_fin=680),
        "aterriza": _mezclar([(0, _nota(150, 0.14, "sin", 0.8, 22, f_fin=55)),
                              (0, _ruido(0.09, 0.35, 30, suave=40))], 0.15),
        "fruta":    _mezclar([(0, _nota(988, 0.10, "tri", 0.45, 10)),
                              (0.07, _nota(1319, 0.22, "tri", 0.45, 7))], 0.3),
        "victoria": _mezclar([(0.00, _nota(_frec(72), 0.25, "tri", 0.4, 4)),
                              (0.13, _nota(_frec(76), 0.25, "tri", 0.4, 4)),
                              (0.26, _nota(_frec(79), 0.25, "tri", 0.4, 4)),
                              (0.39, _nota(_frec(84), 0.9, "tri", 0.4, 2.5)),
                              (0.39, _nota(_frec(76), 0.9, "tri", 0.25, 2.5)),
                              (0.39, _nota(_frec(79), 0.9, "tri", 0.25, 2.5))], 1.35),
        "derrota":  _mezclar([(0.00, _nota(_frec(67), 0.30, "tri", 0.4, 4)),
                              (0.22, _nota(_frec(65), 0.30, "tri", 0.4, 4)),
                              (0.44, _nota(_frec(63), 0.30, "tri", 0.4, 4)),
                              (0.66, _nota(_frec(60), 0.70, "tri", 0.4, 3, f_fin=_frec(59)))], 1.4),
        "mover":    _nota(660, 0.06, "sin", 0.4, 25),
        "ok":       _mezclar([(0, _nota(784, 0.08, "tri", 0.4, 14)),
                              (0.06, _nota(1046, 0.14, "tri", 0.4, 10))], 0.22),
        "tick":     _nota(1000, 0.06, "sin", 0.45, 35),
    }
def _pista(bpm, melodia, bajo, acordes, ritmo):
    """Una melodía por compás (8 corcheas; 0 = silencio), bajo, acordes de fondo y ritmo."""
    cor = 60.0 / bpm / 2
    comp = 8 * cor
    partes = []
    for i, (mel, (r, q), ac) in enumerate(zip(melodia, bajo, acordes)):
        t0 = i * comp
        for k, m in enumerate(mel):
            if m:
                partes.append((t0 + k * cor, _nota(_frec(m), cor * 1.7, "tri", 0.26, 3.5)))
        for k, b in enumerate((r, q, r, q)):
            partes.append((t0 + k * 2 * cor, _nota(_frec(b), cor * 2, "sin", 0.34, 3.0)))
        for m in ac:
            partes.append((t0, _nota(_frec(m), comp, "sin", 0.05, 1.0, ataque=0.25)))
        if ritmo:
            for k in range(8):
                if k % 4 == 0:
                    partes.append((t0 + k * cor, _nota(110, 0.12, "sin", 0.45, 28, f_fin=45)))
                if k % 2 == 1:
                    partes.append((t0 + k * cor, _ruido(0.04, 0.10, 60)))
    buf = _mezclar(partes, comp * len(acordes))
    buf *= 0.85 / max(1e-6, np.abs(buf).max())
    buf[:100] *= np.linspace(0, 1, 100)
    buf[-300:] *= np.linspace(1, 0, 300)
    return buf
def _musica_juego():
    C, Am, F, G = (60, 64, 67), (57, 60, 64), (57, 60, 65), (59, 62, 67)
    mel = [[72, 0, 76, 79, 76, 0, 72, 0], [69, 0, 72, 76, 72, 0, 69, 0],
           [65, 0, 69, 72, 69, 0, 72, 77], [67, 71, 74, 0, 71, 0, 67, 0],
           [76, 79, 84, 79, 76, 0, 79, 0], [81, 76, 72, 76, 69, 0, 72, 0],
           [77, 0, 72, 69, 72, 77, 0, 72], [74, 0, 71, 67, 71, 74, 79, 0]]
    bajo = [(48, 55), (45, 52), (41, 48), (43, 50)] * 2
    return _pista(116, mel, bajo, [C, Am, F, G] * 2, True)
def _musica_menu():
    Am, F, C, G = (57, 60, 64), (53, 57, 60), (55, 60, 64), (55, 59, 62)
    mel = [[76, 0, 72, 0, 69, 0, 72, 76], [77, 0, 76, 72, 69, 0, 0, 0],
           [72, 0, 76, 0, 79, 0, 76, 0], [74, 0, 71, 0, 67, 0, 0, 0],
           [76, 0, 79, 0, 81, 0, 79, 76], [77, 0, 81, 0, 77, 0, 72, 0],
           [74, 0, 79, 0, 74, 0, 71, 0], [72, 0, 76, 0, 79, 0, 0, 0]]
    bajo = [(45, 52), (41, 48), (48, 55), (43, 50), (45, 52), (41, 48), (43, 50), (48, 55)]
    return _pista(84, mel, bajo, [Am, F, C, G, Am, F, G, C], False)
def _a_sonido(w, canales):
    a = (np.clip(w, -1, 1) * 32767 * 0.9).astype(np.int16)
    if canales > 1:
        a = np.column_stack([a] * canales)
    return pygame.sndarray.make_sound(np.ascontiguousarray(a))
class Sonido:
    """Música de fondo (canal 0 reservado) y efectos. Si no hay audio, el juego sigue en silencio."""
    def __init__(self):
        global SR
        self.ok = False
        self.mudo = False
        self.pista = None
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(SR, -16, 2, 512)
            SR, _, canales = pygame.mixer.get_init()
            self.fx = {n: _a_sonido(w, canales) for n, w in _efectos().items()}
            for s in self.fx.values():
                s.set_volume(VOL_EFECTOS)
            self.pistas = {"menu": _a_sonido(_musica_menu(), canales),
                           "juego": _a_sonido(_musica_juego(), canales)}
            pygame.mixer.set_reserved(1)
            self.canal = pygame.mixer.Channel(0)
            self.ok = True
        except Exception as ex:
            print("Sin sonido:", ex)
    def efecto(self, nombre, vol=1.0):
        if self.ok and not self.mudo:
            ch = self.fx[nombre].play()
            if ch:
                ch.set_volume(vol)
    def musica(self, nombre):
        if not self.ok or nombre == self.pista:
            return
        self.pista = nombre
        if nombre is None:
            self.canal.fadeout(500)
        else:
            self.canal.play(self.pistas[nombre], loops=-1, fade_ms=800)
            self.canal.set_volume(0 if self.mudo else VOL_MUSICA)
    def alternar(self):
        self.mudo = not self.mudo
        if self.ok:
            self.canal.set_volume(0 if self.mudo else VOL_MUSICA)
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
        self.salto_ev = self.aterrizo = False
        estaba = self.en_suelo
        if dirx:
            self.mira = dirx
            self.fase += 0.3
        if salto and self.en_suelo:
            self.vy = FUERZA_SALTO
            self.en_suelo = False
            self.salto_ev = True
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
                self.aterrizo = (not estaba) and self.vy > 3
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
#   Cada sprite se dibuja 4 veces más grande, con contorno, brillo en los
#   ojos y mejillas, y se reduce con suavizado (queda sin dientes de sierra).
#   Se guarda en caché por animal, cuadro de animación, pose y tamaño.
# ----------------------------------------------------------------
SS = 4                      # factor de supersampling
MARG_X, MARG_Y = 14, 16     # margen lógico alrededor del rect (orejas, cola, pico...)
LIENZO = (34 + 2 * MARG_X, 40 + MARG_Y + 5)
CUADROS = 16
ROSA = (244, 168, 178)
TINTA = (64, 48, 58)
_CACHE_ANIMALES = {}
def mezcla(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
class Pincel:
    """Dibuja en coordenadas lógicas: (0, 0) es la esquina del rect del animal."""
    def __init__(self, surf):
        self.s = surf
        self.dy = 0.0
    def _p(self, x, y):
        return int(round((x + MARG_X) * SS)), int(round((y + self.dy + MARG_Y) * SS))
    def _r(self, x, y, w, h):
        px, py = self._p(x, y)
        return pygame.Rect(px, py, int(round(w * SS)), int(round(h * SS)))
    def circ(self, col, x, y, r, borde=None, g=0.9):
        c = self._p(x, y)
        if borde:
            pygame.draw.circle(self.s, borde, c, int(round((r + g) * SS)))
        pygame.draw.circle(self.s, col, c, max(1, int(round(r * SS))))
    def elip(self, col, x, y, w, h, borde=None, g=0.9):
        if borde:
            pygame.draw.ellipse(self.s, borde, self._r(x - g, y - g, w + 2 * g, h + 2 * g))
        pygame.draw.ellipse(self.s, col, self._r(x, y, w, h))
    def caja(self, col, x, y, w, h, rad, borde=None, g=0.9):
        if borde:
            pygame.draw.rect(self.s, borde, self._r(x - g, y - g, w + 2 * g, h + 2 * g),
                             border_radius=int((rad + g) * SS))
        pygame.draw.rect(self.s, col, self._r(x, y, w, h), border_radius=int(rad * SS))
    def poli(self, col, pts, borde=None, g=0.9):
        q = [self._p(x, y) for x, y in pts]
        pygame.draw.polygon(self.s, col, q)
        if borde:
            pygame.draw.polygon(self.s, borde, q, max(1, int(g * 1.6 * SS)))
    def lin(self, col, a, b, w=0.8):
        pygame.draw.line(self.s, col, self._p(*a), self._p(*b), max(1, int(w * SS)))
    def arco(self, col, x, y, w, h, a0, a1, wd=0.7):
        pygame.draw.arc(self.s, col, self._r(x, y, w, h), a0, a1, max(1, int(wd * SS)))
    def cadena(self, col, borde, pts):
        """Círculos (x, y, r) unidos: primero todos los contornos y luego los rellenos."""
        for x, y, r in pts:
            self.circ(borde, x, y, r + 0.9)
        for x, y, r in pts:
            self.circ(col, x, y, r)
def _ojo(p, x, y, r=3.2):
    p.circ((255, 255, 255), x, y, r, TINTA, 0.6)
    p.circ(TINTA, x + 0.8, y + 0.4, r * 0.62)
    p.circ((255, 255, 255), x + 1.5, y - 0.6, r * 0.25)
def _patas(p, col, borde, fase, aire, lista, ancho=7.0, y0=28.0):
    """Patas que caminan (alternadas) o se recogen en el aire. Devuelve la posición de los pies."""
    pies = []
    for x, off, lejos in lista:
        if aire:
            dx, lev = (3.0 if x > 17 else -3.0), 3.0
        else:
            dx = math.sin(fase + off) * 3.2
            lev = max(0.0, math.cos(fase + off)) * 2.4
        c = mezcla(col, (0, 0, 0), 0.14) if lejos else col
        yb = 40 - lev
        p.caja(c, x + dx - ancho / 2, y0, ancho, yb - y0, ancho / 2.2, borde)
        pies.append((x + dx, yb))
    return pies
def _pintar(clave, fase, aire):
    c = COLORES[clave]
    cu, ac = c["cuerpo"], c["acento"]
    borde = mezcla(ac, TINTA, 0.5)
    claro = mezcla(cu, (255, 255, 255), 0.6)
    mej = mezcla(cu, ROSA, 0.55)
    surf = pygame.Surface((LIENZO[0] * SS, LIENZO[1] * SS), pygame.SRCALPHA)
    surf.fill((*borde, 0))      # el borde transparente toma el color del contorno (sin halos)
    p = Pincel(surf)
    cuad = [(12, math.pi, True), (26, 0, True), (7, 0, False), (21, math.pi, False)]
    bob = 0.0 if aire else abs(math.sin(fase)) * 1.1
    onda = math.sin(fase) * 1.8
    if clave == "jaguar":
        p.cadena(cu, borde, [(1, 24, 2.6), (-3, 21, 2.6), (-6.5, 16 + onda * 0.3, 2.6),
                             (-7 + onda * 0.5, 10.5, 2.6)])
        p.circ(ac, -7 + onda * 0.5, 10.5, 2.7)
        _patas(p, cu, borde, fase, aire, cuad)
        p.dy = -bob
        p.elip(cu, 1, 14, 32, 20, borde)
        p.elip(claro, 7, 25, 20, 8)
        for x, y in ((5, 21), (11, 18), (10, 26), (17, 27), (23, 27.5)):
            p.circ(ac, x, y, 3.0)
            p.circ(mezcla(cu, ac, 0.3), x, y, 1.4)
        for x in (17.5, 30):
            p.circ(cu, x, 3 if x < 20 else 3.6, 4.3, borde)
            p.circ(ROSA, x, 3.4 if x < 20 else 4, 2.0)
        p.circ(cu, 25, 11.5, 11, borde, 1.0)
        for x, y in ((22, 4.5), (26, 3.5), (19.5, 8)):
            p.circ(ac, x, y, 1.1)
        p.elip(claro, 27.5, 11, 10, 8, borde, 0.8)
        p.elip((222, 130, 150), 34.2, 11.8, 3.6, 2.6)
        p.arco(TINTA, 30.5, 12.6, 5.5, 4, 3.5, 5.9, 0.5)
        p.lin(TINTA, (31, 13.5), (38.5, 12.2), 0.5)
        p.lin(TINTA, (31, 14.5), (38, 15.5), 0.5)
        _ojo(p, 28.5, 8, 3.0)
        p.elip(mej, 21.5, 12.5, 4.6, 2.8)
    elif clave == "oso":
        p.circ(ac, 1.5, 23, 3, borde)
        _patas(p, cu, borde, fase, aire, cuad)
        p.dy = -bob
        p.elip(cu, 1, 14, 32, 20, borde)
        p.elip(claro, 14, 21, 14, 13)
        for x, y in ((17, 3.5), (31, 4)):
            p.circ(ac, x, y, 4.4, borde)
            p.circ(claro, x, y + 0.2, 2.2)
        p.circ(cu, 25, 12, 11, borde, 1.0)
        p.elip(claro, 28, 12, 9.5, 8, borde, 0.8)
        p.elip((92, 70, 78), 34.5, 12.4, 4, 3)
        p.circ(claro, 28.8, 8.5, 4.7, borde, 0.7)       # anteojos
        _ojo(p, 28.8, 8.5, 2.6)
        p.arco(claro, 19, 2, 11, 8, 0.5, 2.6, 1.0)
        p.elip(mej, 22, 14, 4.4, 2.8)
    elif clave == "perezoso":
        for x in (5, 10, 15, 20, 25, 30):                # pelaje desgreñado
            p.poli(cu, [(x - 2.6, 18), (x + (x % 3) - 1, 12.5), (x + 2.6, 18)], borde)
        for x in (7, 13, 19, 25):
            p.poli(cu, [(x - 2.6, 31), (x, 36.5), (x + 2.6, 31)], borde)
        pies = _patas(p, cu, borde, fase, aire, cuad)
        for px, yb in (pies[1], pies[3]):                # garras largas
            for k in (-2, 0, 2):
                p.lin(borde, (px + k * 0.8, yb - 1), (px + k * 1.1, yb + 2.6), 1.7)
                p.lin((240, 228, 204), (px + k * 0.8, yb - 1), (px + k * 1.1, yb + 2.4), 0.9)
        p.dy = -bob
        p.elip(cu, 1, 14, 32, 20, borde)
        p.elip(claro, 7, 25, 20, 8)
        p.circ(mezcla(cu, claro, 0.35), 24, 11, 10.5, borde, 1.0)
        p.elip(claro, 23, 5, 13, 13)
        p.elip(ac, 23.5, 6.5, 10.5, 7)                   # antifaz
        _ojo(p, 28.5, 10, 2.4)
        p.elip(TINTA, 33.5, 11.8, 3.2, 2.4)
        p.arco(TINTA, 28, 12.5, 7, 4.5, 3.5, 6.0, 0.6)
        p.elip(mej, 22.5, 13, 4.2, 2.6)
    elif clave == "chiguiro":
        _patas(p, cu, borde, fase, aire, cuad)
        p.dy = -bob
        p.elip(cu, 0, 13, 33, 22, borde)
        p.elip(claro, 5, 26, 21, 8)
        for x, y in ((6, 19), (11, 17.5), (16, 18.5)):
            p.lin(ac, (x, y), (x + 3, y - 0.8), 0.8)
        p.caja(cu, 17, 4, 19, 17, 7.5, borde)
        p.circ(ac, 20.5, 4.5, 2.4, borde)
        p.elip(mezcla(cu, ac, 0.35), 28.5, 6, 8, 12)
        p.elip(TINTA, 33.2, 7.4, 2, 2.8)
        p.circ(TINTA, 25.8, 8.6, 1.9)
        p.circ((255, 255, 255), 26.4, 8, 0.6)
        p.lin(borde, (36, 16.2), (31.5, 17.2), 0.7)
        p.elip(mej, 22.5, 14, 4.2, 2.6)
    elif clave == "condor":
        p.poli(ac, [(5, 22), (-6, 25), (-5, 30), (3, 29)], borde)    # cola
        p.poli(ac, [(6, 20), (-5, 19), (-6, 24), (4, 25)], borde)
        _patas(p, (240, 208, 120), borde, fase, aire, [(13, math.pi, True), (21, 0, False)],
               ancho=3.6, y0=30)
        p.dy = -bob
        p.elip(cu, 1, 15, 31, 19, borde)
        p.elip(claro, 8, 24, 17, 8)
        wy = 8 if aire else 18                                         # ala: abierta en el aire
        p.poli(ac, [(3, wy + 7), (-5, wy + 12), (7, wy + 11)], borde)
        p.poli(ac, [(5, wy + 9.5), (-2, wy + 14.5), (10, wy + 12.5)], borde)
        p.elip(ac, 2, wy, 22, 13, borde)
        p.cadena((255, 255, 255), borde, [(19.5, 14.5, 4), (25, 16.5, 4.2), (30.5, 15, 4), (32.5, 11.5, 3)])
        p.elip((224, 138, 150), 24, 0.5, 7, 3.6, borde)               # cresta
        p.circ((240, 206, 206), 27.5, 8, 7.5, borde)
        p.poli((255, 226, 130), [(33, 5), (39, 6.2), (42, 10.2), (38.2, 9.2), (33.5, 11)], borde)
        _ojo(p, 29.5, 6.5, 2.4)
    elif clave == "rana":
        p.dy = 0
        _patas(p, cu, borde, fase, aire, [(27, math.pi, True), (21, 0, False)], ancho=5, y0=30)
        pata = mezcla(cu, ac, 0.45)
        fx = -2 + (-3 if aire else math.sin(fase) * 2)
        fy = 32 if aire else 35
        p.elip(pata, fx, fy, 12, 5, borde)                              # pie trasero
        p.dy = -bob
        p.elip(cu, 1, 17, 32, 18, borde)
        p.elip(claro, 8, 26, 19, 8)
        for x, y in ((7, 23), (12, 21), (7.5, 28), (14, 28)):
            p.circ(ac, x, y, 2.1)
        p.elip(pata, 2, 22, 13, 14, borde)                              # muslo
        p.elip(cu, 13, 9, 23, 17, borde)
        for x, y in ((21, 8), (30, 7)):
            p.circ(cu, x, y, 5.2, borde)
        _ojo(p, 21.5, 7.8, 3.3)
        _ojo(p, 30.5, 6.8, 3.3)
        p.arco(borde, 17, 11.5, 19, 9.5, 3.5, 5.95, 0.7)
        p.elip(mezcla(cu, ROSA, 0.6), 29, 16, 4.6, 2.6)
        p.circ(borde, 34.5, 12, 0.7)
    return surf
def dibujar_animal(s, clave, rect, mira, fase=0.0, aire=False):
    k = rect.w / 34
    q = int((fase % math.tau) / math.tau * CUADROS) % CUADROS
    llave = (clave, q, bool(aire), rect.w, mira >= 0)
    img = _CACHE_ANIMALES.get(llave)
    if img is None:
        grande = _pintar(clave, q * math.tau / CUADROS, aire)
        tam = (max(1, round(LIENZO[0] * k)), max(1, round(LIENZO[1] * k)))
        img = pygame.transform.smoothscale(grande, tam)
        if mira < 0:
            img = pygame.transform.flip(img, True, False)
        _CACHE_ANIMALES[llave] = img
    s.blit(img, (rect.x - MARG_X * k, rect.y - MARG_Y * k))
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
        pygame.mixer.pre_init(SR, -16, 2, 512)
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
        self.sonido = Sonido()
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
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_m and self.estado != "nombres":
                    self.sonido.alternar()
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
            self.sonido.musica(self.musica_deseada())
            pygame.display.flip()
            self.reloj.tick(FPS)
    def musica_deseada(self):
        if self.estado == "jugando":
            return "juego" if self.ganador is None else None
        return "menu"
    # ---------- MENÚ ----------
    def elegir_modo(self):
        _, humanos, pcs = MODOS[self.modo]
        if pcs:
            self.sonido.efecto("ok")
            self.estado = "dificultad"
        else:
            self.cfg = DIFICULTADES[1]
            self.iniciar_nombres()
    def menu(self, eventos):
        for e in eventos:
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_UP, pygame.K_w):
                    self.modo = (self.modo - 1) % 3
                    self.sonido.efecto("mover")
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    self.modo = (self.modo + 1) % 3
                    self.sonido.efecto("mover")
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
        self.ayuda("5 niveles · Frutas +1 · Bandera +5   |   R: ranking  ·  M: sonido", 580)
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
                self.sonido.efecto("mover")
            elif e.key in (pygame.K_DOWN, pygame.K_s):
                self.dif = (self.dif + 1) % 3
                self.sonido.efecto("mover")
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
        self.sonido.efecto("ok")
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
                    self.sonido.efecto("mover")
                elif e.key in der:
                    self.cursor[i] = (self.cursor[i] + 1) % len(claves)
                    self.sonido.efecto("mover")
                elif e.key in ok:
                    otros = [self.cursor[k] for k in range(humanos) if k != i and self.listo[k]]
                    if self.cursor[i] not in otros:
                        self.listo[i] = True
                        self.sonido.efecto("ok")
            if e.key == pygame.K_RETURN and all(self.listo):
                self.sonido.efecto("ok")
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
            if j.salto_ev:
                self.sonido.efecto("salto", 0.45 if j.es_pc else 1.0)
            if j.aterrizo:
                self.sonido.efecto("aterriza", 0.35 if j.es_pc else 0.8)
            r = j.rect
            for fr in self.frutas[:]:
                if r.colliderect(fr):
                    self.frutas.remove(fr)
                    j.frutas += 1
                    j.pts_nivel += PUNTOS_FRUTA
                    self.sonido.efecto("fruta", 0.45 if j.es_pc else 1.0)
                    self.chispas(fr.centerx, fr.centery, NIVELES[self.nivel]["fruta"])
            if not congelado and r.colliderect(self.bandera):
                j.llego = True
                j.pts_nivel += PUNTOS_BANDERA
                self.ganador = j
                self.cuenta_final = 110
                self.sonido.efecto("derrota" if j.es_pc else "victoria")
                for _ in range(5):
                    self.chispas(self.bandera.centerx, self.bandera.top, random.choice(
                        [(255, 236, 150), (160, 190, 238), (242, 160, 164)]), 14)
        if self.ganador is None:
            self.tiempo -= 1
            if 0 < self.tiempo <= 10 * FPS and self.tiempo % FPS == 0:
                self.sonido.efecto("tick")
            if self.tiempo <= 0:
                self.cuenta_final = 1
                self.ganador = False  # tiempo agotado
                self.sonido.efecto("derrota")
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
            dibujar_animal(s, j.animal, j.rect, j.mira, j.fase, not j.en_suelo)
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
                self.sonido.efecto("ok")
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
