# -*- coding: utf-8 -*-
"""
==========================================================================
  AVENTURA SALVAJE COLOMBIA 3D
  Videojuego 3D de plataformas con animales y paisajes de Colombia
==========================================================================

INSTALACION (una sola vez):
    pip install ursina pandas scikit-learn

EJECUCION:
    python aventura_salvaje_colombia_3d.py

MENU PRINCIPAL:
    1. PC vs Usuario              -> tu (Jaguar) contra el PC (Titi cabeciblanco)
    2. PC vs Usuario vs Usuario   -> Jaguar + Oso de anteojos + PC
    3. Usuario vs Usuario         -> Jaguar contra Oso de anteojos
    4. Ranking                    -> mejores exploradores (historico guardado)

  Al elegir un modo, el juego te pide tu NOMBRE (y el del jugador 2 si lo hay).
  Al terminar la aventura, tus puntos se guardan en el ranking historico
  (archivo ranking_aventura_salvaje.csv).

CONTROLES:
    Jugador 1 : W A S D  + ESPACIO
    Jugador 2 : Flechas  + CTRL derecho / SHIFT derecho / ENTER
    (En el modo 1 puedes usar WASD o las flechas)
    R = reiniciar nivel   ·   ESC = volver al menu   ·   ENTER = continuar

NIVELES (5 ecosistemas de Colombia):
    1. Selva Amazonica   2. Valle del Cocora   3. Paramo andino
    4. Costa Caribe      5. Llanos Orientales

COMO SE GANA:
    Carrera de plataformas hasta la bandera de Colombia. Se suman puntos por
    las frutas recogidas (1 c/u) y por el orden de llegada (5, 3 y 1 puntos).
    Gana quien tenga mas puntos al terminar los 5 niveles.

LIBRERIAS:
    - ursina        : motor 3D
    - pandas        : tabla de fauna, historial y ranking (guarda CSV)
    - scikit-learn  : el "cerebro" del PC es un arbol de decision entrenado
                      al iniciar cada nivel (decide cuando saltar)
==========================================================================
"""
import sys
import os
import traceback


def _pausa():
    """Evita que la ventana de consola se cierre sola para poder leer el error."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdin.readline()
    except Exception:
        pass


def _falta(modulo, paquete):
    print('\n[ERROR] Falta la libreria "%s".' % modulo)
    print('Instalala con:   python -m pip install %s' % paquete)
    print('Python en uso:   %s (%s)' % (sys.executable, sys.version.split()[0]))
    print('\nPulsa ENTER para cerrar...')
    _pausa()
    sys.exit(1)


try:
    from ursina import *
except ImportError:
    _falta('ursina', 'ursina')
import math
import random
import unicodedata
import datetime
try:
    import numpy as np
except ImportError:
    _falta('numpy', 'numpy')
try:
    import pandas as pd
except ImportError:
    _falta('pandas', 'pandas')
try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    DecisionTreeClassifier = None   # el PC usara reglas simples si no hay scikit-learn

# ------------------------------------------------------------------ AJUSTES
USAR_ACENTOS = True          # pon False si ves cuadritos en lugar de tildes
USAR_SOMBREADO = True        # pon False si los colores se ven muy oscuros
ARCHIVO_RESULTADOS = 'resultados_aventura_salvaje.csv'   # detalle de cada partida
ARCHIVO_RANKING = 'ranking_aventura_salvaje.csv'         # ranking historico
MAX_NOMBRE = 12              # maximo de caracteres del nombre
TOP_RANKING = 10             # cuantos jugadores se muestran en el ranking

FLOOR_Y = -12.0              # altura del suelo / agua lejano
MUERTE_Y = -9.0              # si caes por debajo, reapareces
GRAV = 28.0
SALTO = 11.5
VEL_HUMANO = 8.0
BONUS = {1: 5, 2: 3, 3: 1}   # puntos por orden de llegada
LIMITE_NIVEL = 180.0         # segundos maximos por nivel

SHADER = None
if USAR_SOMBREADO:
    try:
        from ursina.shaders import basic_lighting_shader
        SHADER = basic_lighting_shader
    except Exception:
        SHADER = None


# ------------------------------------------------------------------ UTILES
def C(h):
    """Color a partir de texto hexadecimal (#RRGGBB o #RRGGBBAA)."""
    return color.hex(h)


def cl(v, a, b):
    return a if v < a else (b if v > b else v)


def lr(a, b, t):
    return a + (b - a) * t


def tx(s):
    """Quita tildes si USAR_ACENTOS es False."""
    if USAR_ACENTOS:
        return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))


def transp(e):
    try:
        e.setTransparency(True)
    except Exception:
        pass


def caja(parent, pos, esc, col, modelo='cube', **kw):
    if SHADER is not None and 'shader' not in kw:
        kw['shader'] = SHADER
    return Entity(parent=parent, model=modelo, position=pos, scale=esc, color=col, **kw)


# ------------------------------------------------------------------ DATOS (pandas)
FAUNA = pd.DataFrame([
    (1, 'Selva Amazónica', 'Jaguar', 'Panthera onca', 'El felino más grande de América'),
    (1, 'Selva Amazónica', 'Guacamaya roja', 'Ara macao', 'Ave de colores vivos que vuela en pareja'),
    (1, 'Selva Amazónica', 'Delfín rosado', 'Inia geoffrensis', 'Delfín de río que nada entre la selva inundada'),
    (2, 'Valle del Cocora', 'Loro orejiamarillo', 'Ognorhynchus icterotis', 'Anida en las palmas de cera; está amenazado'),
    (2, 'Valle del Cocora', 'Oso de anteojos', 'Tremarctos ornatus', 'El único oso de Suramérica'),
    (2, 'Valle del Cocora', 'Palma de cera', 'Ceroxylon quindiuense', 'Árbol nacional de Colombia, llega a 60 m'),
    (3, 'Páramo andino', 'Cóndor de los Andes', 'Vultur gryphus', 'Una de las aves voladoras más grandes del mundo'),
    (3, 'Páramo andino', 'Tapir de montaña', 'Tapirus pinchaque', 'Habitante tímido de los páramos'),
    (3, 'Páramo andino', 'Frailejón', 'Espeletia', 'Planta que atrapa el agua de la neblina'),
    (4, 'Costa Caribe', 'Tití cabeciblanco', 'Saguinus oedipus', 'Pequeño primate del norte de Colombia'),
    (4, 'Costa Caribe', 'Flamenco rosado', 'Phoenicopterus ruber', 'Colonias enormes en La Guajira'),
    (4, 'Costa Caribe', 'Palma de coco', 'Cocos nucifera', 'Sombra de todas las playas del Caribe'),
    (5, 'Llanos Orientales', 'Chigüiro', 'Hydrochoerus hydrochaeris', 'El roedor más grande del mundo'),
    (5, 'Llanos Orientales', 'Oso hormiguero', 'Myrmecophaga tridactyla', 'Come miles de hormigas al día'),
    (5, 'Llanos Orientales', 'Caimán del Orinoco', 'Crocodylus intermedius', 'Uno de los reptiles más amenazados'),
], columns=['nivel', 'ecosistema', 'animal', 'cientifico', 'dato'])

NIVELES = [
    dict(nombre='Selva Amazónica', sub='Entre lianas, ríos y guacamayas',
         sky='#c9f1e4', piso='#86d9c0', top='#b8eaa3', side='#8ccf92',
         tronco='#d2ab82', hojas=['#8fdc9f', '#b5f0a6', '#74cf9b'], colina='#a4e3bb',
         fruta='#ffb347', deco='selva', ave=('#ff8f8f', '#7fc4ff', 1.0),
         sol='#fff3b0', fauna_plat='rana'),
    dict(nombre='Valle del Cocora', sub='Palmas de cera entre la neblina del Eje Cafetero',
         sky='#d3ebff', piso='#a9dba0', top='#c6ecaa', side='#9fd08c',
         tronco='#efe2c2', hojas=['#8fdcaa', '#a8e8b8', '#7ccfa0'], colina='#bde5b5',
         fruta='#ff8f8f', deco='cera', ave=('#9be58f', '#ffe56b', 0.9),
         sol='#fff6c4', fauna_plat=None),
    dict(nombre='Páramo andino', sub='Frailejones y cóndores sobre las nubes',
         sky='#e6dcff', piso='#a8c9f2', top='#dcecc4', side='#b8d1a8',
         tronco='#d2c59d', hojas=['#e9f3cb', '#d7ebc0'], colina='#cfd8f0',
         fruta='#ffd36e', deco='frailejon', ave=('#8c93b8', '#aeb4d6', 1.8),
         sol='#fff0d0', fauna_plat=None),
    dict(nombre='Costa Caribe', sub='Playas, palmas de coco y flamencos',
         sky='#c2efff', piso='#7fe0ea', top='#ffecbf', side='#f2d196',
         tronco='#dab68b', hojas=['#8ce3ab', '#a9efb9'], colina='#ffdcc4',
         fruta='#ffe066', deco='coco', ave=('#ffb3c9', '#ffc9d9', 1.0),
         sol='#fff7b8', fauna_plat=None),
    dict(nombre='Llanos Orientales', sub='Sabana infinita al atardecer',
         sky='#ffdccb', piso='#ffd0a0', top='#ebf4ad', side='#cbd98e',
         tronco='#d9b08c', hojas=['#9ad99a', '#b9e6a0'], colina='#ffbfae',
         fruta='#ff9ec7', deco='saman', ave=('#ffffff', '#e8f0ff', 1.1),
         sol='#ff9f8a', fauna_plat='chiguiro'),
]

MOV_X = ('#ffd1dc', '#e7a8bd')   # plataformas que se mueven de lado (rosadas)
MOV_Y = ('#d8cbff', '#b7a6ea')   # plataformas que suben y bajan (lilas)

# ------------------------------------------------------------------ APP
app = Ursina(title='Aventura Salvaje Colombia 3D', borderless=False)
for _ajuste in (lambda: setattr(window, 'fullscreen', False),
                lambda: setattr(window.exit_button, 'visible', False),
                lambda: setattr(window.fps_counter, 'enabled', False),
                lambda: setattr(camera, 'clip_plane_far', 1500),
                lambda: setattr(camera, 'fov', 72)):
    try:
        _ajuste()
    except Exception:
        pass

G = dict(
    estado='menu', modo=0, nivel=0, jug=[], plats=[], frutas=[], root=None,
    aves=[], nubes=[], banderas=[], sol=None, t=0.0, t_nivel=0.0, t_intro=0.0,
    llegadas=0, cuenta_final=None, hist=[], hud_t=0.0, clf=None, rng=random.Random(7),
    # interfaz
    ui=[], hud=None, ui_cuenta=None, ui_campo=None, cur_ant=None,
    # nombres y ranking
    modo_pend=0, nombres=[], nom_buf='', ultimos=[], recien=[],
)


# ------------------------------------------------------------------ PLATAFORMAS
class Plataforma:
    def __init__(s, idx, x, top, z, w, d, cfg, rng, parent, tipo='fija', meta=False):
        s.idx = idx
        s.bx = s.x = x
        s.bt = s.top = top
        s.bz = s.z = z
        s.px, s.pt = x, top
        s.dx = s.dy = 0.0
        s.w, s.d, s.th = w, d, 3.0
        s.tipo = tipo
        s.meta = meta
        s.amp = 2.2 if tipo == 'mov_x' else (0.9 if tipo == 'mov_y' else 0.0)
        s.vel = rng.uniform(0.7, 1.0)
        s.fase = rng.uniform(0, 6.28)
        s.root = Entity(parent=parent, position=(x, top, z))
        col_top, col_side = cfg['top'], cfg['side']
        if tipo == 'mov_x':
            col_top, col_side = MOV_X
        elif tipo == 'mov_y':
            col_top, col_side = MOV_Y
        elif meta:
            col_top = '#fff0a6'
        caja(s.root, (0, -s.th / 2 - 0.2, 0), (w, s.th, d), C(col_side))
        caja(s.root, (0, -0.18, 0), (w + 0.12, 0.36, d + 0.12), C(col_top))
        if idx > 0 and not meta:
            s._decorar(cfg, rng)

    def _decorar(s, cfg, rng):
        for _ in range(rng.randint(2, 4)):
            px = rng.choice((-1, 1)) * rng.uniform(0.4, s.w / 2 - 0.6)
            pz = rng.choice((-1, 1)) * rng.uniform(0.4, s.d / 2 - 0.6)
            if cfg['deco'] == 'frailejon':
                caja(s.root, (px, 0.3, pz), (0.22, 0.6, 0.22), C('#cdbf9a'))
                caja(s.root, (px, 0.72, pz), (0.55, 0.45, 0.55), C('#eef6d2'), 'sphere')
            elif cfg['deco'] == 'coco':
                caja(s.root, (px, 0.08, pz), (0.45, 0.14, 0.45), C(rng.choice(['#ffd6e0', '#ffffff', '#ffe0b8'])), 'sphere')
            else:
                caja(s.root, (px, 0.14, pz), (0.28, 0.28, 0.28),
                     C(rng.choice(['#ff9ebb', '#fff07a', '#ffffff', '#b9a7ff'])), 'sphere')
        if cfg['fauna_plat'] and s.tipo == 'fija' and rng.random() < 0.28:
            px = rng.choice((-1, 1)) * (s.w / 2 - 1.0)
            pz = rng.uniform(-s.d / 2 + 1.0, s.d / 2 - 1.0)
            if cfg['fauna_plat'] == 'rana':
                rana(s.root, px, pz)
            else:
                chiguiro(s.root, px, pz)

    def actualizar(s, t):
        s.px, s.pt = s.x, s.top
        if s.tipo == 'mov_x':
            s.x = s.bx + math.sin(t * s.vel + s.fase) * s.amp
        elif s.tipo == 'mov_y':
            s.top = s.bt + math.sin(t * s.vel + s.fase) * s.amp
        s.dx = s.x - s.px
        s.dy = s.top - s.pt
        s.root.position = (s.x, s.top, s.z)


def rana(parent, x, z):
    e = Entity(parent=parent, position=(x, 0, z), rotation_y=random.uniform(0, 360))
    caja(e, (0, 0.22, 0), (0.55, 0.36, 0.7), C('#9be58f'), 'sphere')
    for lado in (-1, 1):
        caja(e, (0.16 * lado, 0.45, 0.2), (0.2, 0.2, 0.2), C('#ffffff'), 'sphere')
        caja(e, (0.16 * lado, 0.47, 0.28), (0.09, 0.09, 0.09), C('#3b3b52'), 'sphere')
    caja(e, (0, 0.08, 0.0), (0.7, 0.1, 0.9), C('#7fd37e'), 'sphere')


def chiguiro(parent, x, z):
    e = Entity(parent=parent, position=(x, 0, z), rotation_y=random.uniform(0, 360))
    caja(e, (0, 0.45, 0), (0.7, 0.55, 1.1), C('#d9b08a'))
    caja(e, (0, 0.62, 0.7), (0.5, 0.45, 0.5), C('#cfa57e'))
    caja(e, (0, 0.55, 0.98), (0.3, 0.25, 0.12), C('#b88a68'))
    for sx in (-1, 1):
        caja(e, (0.2 * sx, 0.18, 0.35), (0.16, 0.36, 0.16), C('#c39a74'))
        caja(e, (0.2 * sx, 0.18, -0.35), (0.16, 0.36, 0.16), C('#c39a74'))
        caja(e, (0.16 * sx, 0.9, 0.6), (0.1, 0.12, 0.1), C('#cfa57e'))
        caja(e, (0.17 * sx, 0.7, 0.9), (0.07, 0.07, 0.07), C('#3b3b52'), 'sphere')


class Fruta:
    def __init__(s, x, y, z, col, parent):
        s.x, s.y0, s.z = x, y, z
        s.ent = Entity(parent=parent, position=(x, y, z))
        caja(s.ent, (0, 0, 0), (0.65, 0.65, 0.65), C(col), 'sphere')
        caja(s.ent, (0.06, 0.4, 0), (0.14, 0.22, 0.05), C('#7fdc8f'), rotation_z=25)
        s.activa = True
        s.fase = random.uniform(0, 6.28)

    def animar(s, t):
        s.ent.rotation_y = t * 110
        s.ent.y = s.y0 + math.sin(t * 3 + s.fase) * 0.18


# ------------------------------------------------------------------ ANIMALES
def construir_animal(tipo, cu):
    """Arma el animal con cubos y esferas. Devuelve la lista de patas."""
    patas = []
    if tipo == 'jaguar':
        pel, claro, mancha = '#ffe0a0', '#fff4d8', '#dca36a'
        caja(cu, (0, 0.85, 0), (0.95, 0.75, 1.55), C(pel))
        caja(cu, (0, 0.52, 0.02), (0.99, 0.2, 1.4), C(claro))
        for (mx, mz) in [(-0.25, 0.4), (0.25, 0.1), (-0.2, -0.3), (0.3, -0.5), (-0.3, 0.0), (0.1, 0.6)]:
            caja(cu, (mx, 1.24, mz), (0.2, 0.05, 0.2), C(mancha))
        caja(cu, (0, 1.12, 0.98), (0.82, 0.75, 0.78), C(pel), 'sphere')
        caja(cu, (0, 1.0, 1.36), (0.42, 0.3, 0.32), C(claro))
        caja(cu, (0, 1.06, 1.53), (0.14, 0.1, 0.08), C('#ff9aa8'))
        for sx in (-1, 1):
            caja(cu, (0.23 * sx, 1.26, 1.32), (0.11, 0.11, 0.06), C('#3d3d52'), 'sphere')
            caja(cu, (0.33 * sx, 1.55, 0.88), (0.26, 0.26, 0.2), C(pel), 'sphere')
            caja(cu, (0.33 * sx, 1.55, 0.95), (0.14, 0.14, 0.1), C('#ffc4b0'), 'sphere')
        caja(cu, (0, 1.05, -1.1), (0.2, 0.2, 1.0), C(pel), rotation_x=-28)
        caja(cu, (0, 1.32, -1.5), (0.22, 0.22, 0.25), C(mancha), rotation_x=-28)
        for (px, pz) in [(-0.3, 0.55), (0.3, 0.55), (-0.3, -0.55), (0.3, -0.55)]:
            patas.append(caja(cu, (px, 0.5, pz), (0.3, 0.5, 0.3), C(pel), origin_y=.5))
    elif tipo == 'oso':
        pel, claro = '#b79a8a', '#f7e8d6'
        caja(cu, (0, 0.95, 0), (1.1, 0.95, 1.5), C(pel))
        caja(cu, (0, 0.95, 0.5), (0.8, 0.8, 0.6), C(claro))
        caja(cu, (0, 1.2, 0.95), (0.95, 0.9, 0.85), C(pel), 'sphere')
        caja(cu, (0, 1.02, 1.38), (0.5, 0.4, 0.38), C(claro))
        caja(cu, (0, 1.1, 1.58), (0.17, 0.12, 0.1), C('#6b5b5b'))
        for sx in (-1, 1):
            caja(cu, (0.26 * sx, 1.36, 1.28), (0.34, 0.34, 0.18), C('#fff7e8'), 'sphere')
            caja(cu, (0.26 * sx, 1.36, 1.36), (0.1, 0.1, 0.06), C('#3d3d52'), 'sphere')
            caja(cu, (0.4 * sx, 1.72, 0.82), (0.3, 0.3, 0.24), C(pel), 'sphere')
            caja(cu, (0.4 * sx, 1.72, 0.9), (0.16, 0.16, 0.1), C('#e8c9b8'), 'sphere')
        caja(cu, (0, 0.95, -0.8), (0.25, 0.25, 0.25), C(pel), 'sphere')
        for (px, pz) in [(-0.34, 0.5), (0.34, 0.5), (-0.34, -0.5), (0.34, -0.5)]:
            patas.append(caja(cu, (px, 0.5, pz), (0.38, 0.5, 0.38), C(pel), origin_y=.5))
    else:  # titi cabeciblanco
        pel, claro = '#caa88a', '#fff2e6'
        caja(cu, (0, 0.8, 0), (0.62, 0.65, 0.85), C(pel))
        caja(cu, (0, 0.72, 0.1), (0.5, 0.45, 0.7), C(claro))
        caja(cu, (0, 1.28, 0.4), (0.62, 0.62, 0.62), C(claro), 'sphere')
        caja(cu, (0, 1.22, 0.68), (0.3, 0.28, 0.18), C('#8a7283'), 'sphere')
        for sx in (-1, 1):
            caja(cu, (0.15 * sx, 1.32, 0.68), (0.09, 0.09, 0.06), C('#2f2f42'), 'sphere')
        for k, ang in enumerate((-30, -15, 0, 15, 30)):
            caja(cu, (0.09 * (k - 2), 1.75, 0.36), (0.09, 0.5, 0.09), C('#ffffff'), rotation_z=ang)
        seg = [(0, 0.95, -0.55, 25), (0, 1.15, -0.85, 55), (0, 1.5, -0.95, 85), (0, 1.8, -0.85, 115)]
        for (a, b, c, rx) in seg:
            caja(cu, (a, b, c), (0.14, 0.14, 0.4), C('#b68f74'), rotation_x=-rx + 90)
        for (px, pz) in [(-0.2, 0.3), (0.2, 0.3), (-0.2, -0.3), (0.2, -0.3)]:
            patas.append(caja(cu, (px, 0.5, pz), (0.17, 0.5, 0.17), C(pel), origin_y=.5))
    return patas


# ------------------------------------------------------------------ JUGADOR
class Jugador:
    R = 0.55

    def __init__(s, etq, nombre, animal, es_bot, teclas, col_marc, col_txt, esc=1.0):
        s.etq, s.nombre, s.animal = etq, nombre, animal
        s.es_bot = es_bot
        s.teclas = teclas
        s.col_marc = col_marc
        s.col_txt = col_txt
        s.root = Entity()
        s.cuerpo = Entity(parent=s.root)
        s.patas = construir_animal(animal['tipo'], s.cuerpo)
        s.cuerpo.scale = esc
        s.esc = esc
        s.marc = caja(s.root, (0, 2.5 * esc + 0.4, 0), 0.32, C(col_marc), rotation=(45, 0, 45), shader=None)
        s.sombra = Entity(model='sphere', scale=(1.2, 0.03, 1.2), color=C('#2b2b4a55'))
        transp(s.sombra)
        s.total = 0
        s.rot = 0.0
        s.fase_anim = 0.0
        s.reiniciar(0, 0, 0)

    def reiniciar(s, x, y, z):
        s.x, s.y, s.z = x, y, z
        s.vx = s.vy = s.vz = 0.0
        s.suelo = None
        s.coyote = 0.0
        s.aturdido = 0.0
        s.frutas = 0
        s.llego = False
        s.puesto = 0
        s.pts_nivel = 0
        s.ultima_segura = (x, y + 0.2, z)
        s.idx_actual = 0
        s.nueva_plat = False
        s.t_misma = 0.0
        s.sesgo = 0.0
        s.sincronizar()

    def reaparecer(s):
        s.x, s.y, s.z = s.ultima_segura
        s.vx = s.vy = s.vz = 0.0
        s.suelo = None
        s.aturdido = 0.5

    # ----- fisica propia (rapida y estable)
    def fisica(s, dt, ix, iz, saltar, vmax, plats):
        if s.aturdido > 0:
            s.aturdido -= dt
            ix = iz = 0.0
            saltar = False
        acc = 46.0 if s.suelo is not None else 24.0
        s.vx += cl(ix * vmax - s.vx, -acc * dt, acc * dt)
        s.vz += cl(iz * vmax - s.vz, -acc * dt, acc * dt)
        if s.suelo is not None:
            s.coyote = 0.12
        else:
            s.coyote -= dt
        if saltar and s.coyote > 0 and s.vy <= 0.5:
            s.vy = SALTO
            s.coyote = 0
            s.suelo = None
        s.vy = max(s.vy - GRAV * dt, -38.0)
        ant_y = s.y
        if s.suelo is not None:
            s.x += s.suelo.dx
            s.y += s.suelo.dy
        s.x += s.vx * dt
        s.z += s.vz * dt
        s.y += s.vy * dt

        aterrizo = None
        if s.vy <= 0:
            for p in plats:
                if abs(s.x - p.x) <= p.w / 2 + 0.25 and abs(s.z - p.z) <= p.d / 2 + 0.25:
                    if ant_y >= p.pt - 0.15 and s.y <= p.top + 0.001:
                        if aterrizo is None or p.top > aterrizo.top:
                            aterrizo = p
        if aterrizo is not None:
            s.y = aterrizo.top
            s.vy = 0.0
            s.suelo = aterrizo
            s.ultima_segura = (aterrizo.bx, aterrizo.bt + 0.6, aterrizo.bz)
            if aterrizo.idx != s.idx_actual:
                s.idx_actual = aterrizo.idx
                s.nueva_plat = True
        else:
            s.suelo = None

        # choques laterales con los bloques
        for p in plats:
            if p is aterrizo:
                continue
            base = p.top - p.th - 0.2
            if s.y + 1.1 > base and s.y < p.top - 0.3:
                ox = (p.w / 2 + Jugador.R) - abs(s.x - p.x)
                oz = (p.d / 2 + Jugador.R) - abs(s.z - p.z)
                if ox > 0 and oz > 0:
                    if ox < oz:
                        s.x += math.copysign(ox, s.x - p.x)
                        s.vx = 0.0
                    else:
                        s.z += math.copysign(oz, s.z - p.z)
                        s.vz = 0.0
        if s.y < MUERTE_Y:
            s.reaparecer()

    # ----- cerebro del PC (scikit-learn)
    def ia(s, dt, plats, clf, nivel_idx):
        rng = G['rng']
        if s.nueva_plat:
            s.nueva_plat = False
            s.t_misma = 0.0
            p_fallo = max(0.04, 0.18 - 0.03 * nivel_idx)
            s.sesgo = rng.choice((0.9, -1.2)) if rng.random() < p_fallo else 0.0
        s.t_misma += dt
        cur = s.suelo
        tgt = plats[min(s.idx_actual + 1, len(plats) - 1)]
        dx, dz = tgt.x - s.x, tgt.z - s.z
        dist = math.hypot(dx, dz)
        if dist < 0.01:
            return 0.0, 0.0, False
        ix, iz = dx / dist, dz / dist
        saltar = False
        if cur is not None and cur is not tgt:
            db = dist_borde(s.x, s.z, cur, ix, iz)
            hueco = max(0.0, (tgt.z - tgt.d / 2) - (cur.z + cur.d / 2))
            dh = tgt.top - cur.top
            X = np.array([[db + s.sesgo, hueco, dh, 1.0]])
            saltar = bool(clf.predict(X)[0])
            if s.t_misma > 5.0 and db < 1.4:
                saltar = True
        return ix, iz, saltar

    # ----- visual
    def sincronizar(s):
        s.root.position = (s.x, s.y, s.z)

    def animar(s, dt, plats):
        v = math.hypot(s.vx, s.vz)
        if v > 0.4:
            tgt = math.degrees(math.atan2(s.vx, s.vz))
            d = (tgt - s.rot + 180) % 360 - 180
            s.rot += d * min(1.0, dt * 14)
            s.cuerpo.rotation_y = s.rot
        if s.suelo is not None and v > 0.8:
            s.fase_anim += dt * v * 1.3
            sw = math.sin(s.fase_anim) * 38
            for i, pata in enumerate(s.patas):
                pata.rotation_x = sw if i in (0, 3) else -sw
            s.cuerpo.y = abs(math.sin(s.fase_anim)) * 0.07
        elif s.suelo is None:
            for pata in s.patas:
                pata.rotation_x = lr(pata.rotation_x, -25 if s.vy > 0 else 20, min(1.0, dt * 10))
            s.cuerpo.y = 0
        else:
            for pata in s.patas:
                pata.rotation_x = lr(pata.rotation_x, 0, min(1.0, dt * 12))
            s.cuerpo.y = 0
        s.marc.rotation_y += 140 * dt
        s.marc.y = 2.5 * s.esc + 0.4 + math.sin(G['t'] * 4) * 0.1
        # sombra
        gy = None
        for p in plats:
            if abs(s.x - p.x) <= p.w / 2 and abs(s.z - p.z) <= p.d / 2 and p.top <= s.y + 0.3:
                if gy is None or p.top > gy:
                    gy = p.top
        if gy is None:
            s.sombra.enabled = False
        else:
            s.sombra.enabled = True
            k = cl(1.25 - (s.y - gy) * 0.08, 0.5, 1.25)
            s.sombra.position = (s.x, gy + 0.04, s.z)
            s.sombra.scale = (k, 0.03, k)

    def destruir(s):
        destroy(s.root)
        destroy(s.sombra)


def dist_borde(px, pz, p, ix, iz):
    """Distancia desde (px,pz) hasta el borde de la plataforma en la direccion (ix,iz)."""
    ts = []
    if abs(ix) > 1e-6:
        bx = p.x + math.copysign(p.w / 2, ix)
        ts.append((bx - px) / ix)
    if abs(iz) > 1e-6:
        bz = p.z + math.copysign(p.d / 2, iz)
        ts.append((bz - pz) / iz)
    return max(0.0, min(ts)) if ts else 99.0


# ------------------------------------------------------------------ IA: arbol de decision
COLS_IA = ['dist_borde', 'hueco', 'dif_altura', 'en_suelo']


def entrenar_ia(nivel_idx):
    """Genera datos de ejemplo con pandas y entrena un arbol de decision
    (scikit-learn) que decide si el PC debe saltar o seguir caminando."""
    rng = random.Random(12345 + nivel_idx)
    filas = []
    for _ in range(4000):
        db = rng.uniform(0, 6)
        hueco = rng.uniform(0, 5)
        dh = rng.uniform(-2, 2)
        suelo = 1 if rng.random() < 0.85 else 0
        umbral = rng.gauss(0.9, 0.12)
        if not suelo:
            y = 0
        elif hueco < 0.25 and dh < 0.45:
            y = 0
        elif dh >= 0.45:
            y = 1 if db < umbral + 0.8 else 0
        else:
            y = 1 if db < umbral else 0
        filas.append((db, hueco, dh, suelo, y))
    df = pd.DataFrame(filas, columns=COLS_IA + ['salta'])
    if DecisionTreeClassifier is None:
        return ReglaSalto()
    clf = DecisionTreeClassifier(max_depth=6, random_state=1)
    clf.fit(df[COLS_IA].values, df['salta'].values)
    return clf


class ReglaSalto:
    """Respaldo sin scikit-learn: misma idea que el arbol de decision."""

    def predict(self, X):
        db, hueco, dh, suelo = X[0]
        if not suelo or (hueco < 0.25 and dh < 0.45):
            return [0]
        return [1 if db < (1.7 if dh >= 0.45 else 0.9) else 0]


# ------------------------------------------------------------------ GENERACION DE NIVELES
def generar_pista(n, cfg, rng, root):
    L = n + 1
    plats = [Plataforma(0, 0, 0, 0, 12, 9, cfg, rng, root)]
    x = top = z = 0.0
    d_prev = 9.0
    total = 9 + 3 * L
    prev_mov_y = False
    for i in range(1, total + 1):
        meta = (i == total)
        w = 10.0 if meta else rng.uniform(5.5 - 0.35 * L, 7.2 - 0.35 * L)
        d = 10.0 if meta else rng.uniform(4.6 - 0.2 * L, 6.0 - 0.2 * L)
        gap = rng.uniform(1.0, 1.6 + 0.45 * L)
        z = z + d_prev / 2 + gap + d / 2
        x = cl(x + rng.uniform(-1, 1) * (2.0 + 0.9 * L), -9, 9)
        tipo = 'fija'
        if not meta and L >= 3 and i % 3 == 0:
            tipo = 'mov_x'
        if not meta and L >= 4 and i % 4 == 0:
            tipo = 'mov_y'
        if prev_mov_y or tipo == 'mov_y' or meta:
            dh = rng.choice([-1.0, 0.0, 0.0, 0.5])
        else:
            dh = rng.choice([-1.0, 0.0, 0.0, 0.6, 1.0, 1.4])
        top = cl(top + dh, 0, 7)
        plats.append(Plataforma(i, x, top, z, w, d, cfg, rng, root, tipo, meta))
        prev_mov_y = (tipo == 'mov_y')
        d_prev = d
    return plats


def crear_frutas(plats, cfg, rng, root):
    frutas = []
    for i, p in enumerate(plats):
        if p.tipo == 'fija' and 0 < i < len(plats) - 1 and rng.random() < 0.6:
            fx = p.x + rng.uniform(-p.w / 2 + 1, p.w / 2 - 1)
            fz = p.z + rng.uniform(-p.d / 2 + 1, p.d / 2 - 1)
            frutas.append(Fruta(fx, p.top + 1.1, fz, cfg['fruta'], root))
        if i < len(plats) - 1 and rng.random() < 0.5:
            q = plats[i + 1]
            mx, mz = (p.x + q.x) / 2, (p.z + q.z) / 2
            frutas.append(Fruta(mx, max(p.top, q.top) + 2.3, mz, cfg['fruta'], root))
    return frutas


def crear_meta(plat, root):
    pm = plat.root
    caja(pm, (0, 3.5, 0), (0.25, 7, 0.25), C('#f5f5f5'))
    caja(pm, (0, 7.1, 0), (0.45, 0.45, 0.45), C('#ffe066'), 'sphere')
    tela = Entity(parent=pm, position=(0, 6.4, 0))
    caja(tela, (1.3, 0.25, 0), (2.4, 0.5, 0.08), C('#ffe066'))
    caja(tela, (1.3, -0.125, 0), (2.4, 0.25, 0.08), C('#7aa8ff'))
    caja(tela, (1.3, -0.375, 0), (2.4, 0.25, 0.08), C('#ff7a7a'))
    G['banderas'].append(tela)


def arbol_selva(root, x, z, h, cfg, rng):
    caja(root, (x, FLOOR_Y + h / 2, z), (h * 0.12, h, h * 0.12), C(cfg['tronco']))
    for _ in range(3):
        r = h * rng.uniform(0.28, 0.4)
        caja(root, (x + rng.uniform(-r * .5, r * .5), FLOOR_Y + h + rng.uniform(-.1, .2) * h, z + rng.uniform(-r * .5, r * .5)),
             (r * 2, r * 1.6, r * 2), C(rng.choice(cfg['hojas'])), 'sphere')


def palma_cera(root, x, z, h, cfg, rng):
    h *= 1.5
    caja(root, (x, FLOOR_Y + h / 2, z), (h * 0.035, h, h * 0.035), C(cfg['tronco']))
    for _ in range(3):
        r = h * rng.uniform(0.08, 0.12)
        caja(root, (x + rng.uniform(-r, r), FLOOR_Y + h + rng.uniform(-1, 1), z + rng.uniform(-r, r)),
             (r * 2.2, r * 0.8, r * 2.2), C(rng.choice(cfg['hojas'])), 'sphere')


def frailejon_gigante(root, x, z, h, cfg, rng):
    h *= 0.6
    caja(root, (x, FLOOR_Y + h / 2, z), (h * 0.18, h, h * 0.18), C(cfg['tronco']))
    caja(root, (x, FLOOR_Y + h + h * 0.1, z), (h * 0.7, h * 0.55, h * 0.7), C(rng.choice(cfg['hojas'])), 'sphere')


def palma_coco(root, x, z, h, cfg, rng):
    h *= 0.9
    caja(root, (x, FLOOR_Y + h / 2, z), (h * 0.06, h, h * 0.06), C(cfg['tronco']), rotation_z=rng.uniform(-6, 6))
    copa = Entity(parent=root, position=(x, FLOOR_Y + h, z))
    for k in range(6):
        piv = Entity(parent=copa, rotation_y=k * 60)
        caja(piv, (0, 0, h * 0.16), (h * 0.07, h * 0.015, h * 0.34), C(rng.choice(cfg['hojas'])), rotation_x=22)


def saman(root, x, z, h, cfg, rng):
    h *= 0.7
    caja(root, (x, FLOOR_Y + h / 2, z), (h * 0.09, h, h * 0.09), C(cfg['tronco']))
    caja(root, (x, FLOOR_Y + h + 0.5, z), (h * 1.1, h * 0.3, h * 1.1), C(rng.choice(cfg['hojas'])), 'sphere')


def decorar(cfg, rng, root, zmax):
    n = int(max(34, zmax / 3.2))
    tipo = cfg['deco']
    f = dict(selva=arbol_selva, cera=palma_cera, frailejon=frailejon_gigante,
             coco=palma_coco, saman=saman)[tipo]
    for _ in range(n):
        x = rng.choice((-1, 1)) * rng.uniform(15, 44)
        z = rng.uniform(-25, zmax + 45)
        f(root, x, z, rng.uniform(16, 30), cfg, rng)
    # suelo / agua lejano
    caja(root, (0, FLOOR_Y - 0.5, zmax / 2), (420, 1, zmax + 420), C(cfg['piso']))
    # colinas y montañas suaves
    for _ in range(12):
        lado = rng.choice((-1, 1))
        caja(root, (lado * rng.uniform(75, 130), FLOOR_Y + 6, rng.uniform(-40, zmax + 120)),
             (rng.uniform(40, 80), rng.uniform(26, 46), rng.uniform(40, 80)), C(cfg['colina']), 'sphere')
    # nubes
    G['nubes'] = []
    for _ in range(16):
        e = Entity(parent=root, position=(rng.uniform(-90, 90), rng.uniform(28, 55), rng.uniform(-30, zmax + 140)))
        for k in range(3):
            caja(e, (k * 5 - 5, rng.uniform(-0.5, 0.8), rng.uniform(-1, 1)),
                 (rng.uniform(8, 13), rng.uniform(2.5, 4), rng.uniform(5, 8)), C('#ffffffee'), 'sphere')
        G['nubes'].append((e, rng.uniform(0.6, 1.6)))
    # sol
    G['sol'] = caja(root, (0, 0, 0), 50, C(cfg['sol']), 'sphere')


def crear_ave(root, cc, ac, tam):
    e = Entity(parent=root)
    caja(e, (0, 0, 0), (0.5 * tam, 0.4 * tam, 1.1 * tam), C(cc), 'sphere')
    caja(e, (0, 0.1 * tam, 0.6 * tam), (0.32 * tam, 0.32 * tam, 0.32 * tam), C(cc), 'sphere')
    caja(e, (0, 0.05 * tam, 0.85 * tam), (0.1 * tam, 0.08 * tam, 0.25 * tam), C('#ffd27a'))
    caja(e, (0, 0.02 * tam, -0.75 * tam), (0.3 * tam, 0.06 * tam, 0.6 * tam), C(ac))
    alas = []
    for lado in (1, -1):
        alas.append(caja(e, (0.15 * tam * lado, 0.05 * tam, 0), (1.6 * tam, 0.06 * tam, 0.7 * tam),
                         C(ac), origin_x=-0.5 * lado))
    return e, alas


def crear_aves(cfg, rng, root, zmax):
    G['aves'] = []
    cc, ac, tam = cfg['ave']
    n = max(5, int(zmax / 30))
    for i in range(n):
        e, alas = crear_ave(root, cc, ac, tam)
        G['aves'].append(dict(e=e, alas=alas, cx=rng.uniform(-6, 6), cy=rng.uniform(9, 22),
                              cz=(i + 0.5) * zmax / n, rad=rng.uniform(14, 30),
                              vel=rng.uniform(0.25, 0.5) * rng.choice((-1, 1)), fase=rng.uniform(0, 6.28)))


def cargar_nivel(n):
    if G['root'] is not None:
        destroy(G['root'])
    G['banderas'] = []
    G['root'] = root = Entity()
    cfg = NIVELES[n]
    rng = random.Random(500 + n * 17)
    window.color = C(cfg['sky'])
    try:
        scene.fog_color = C(cfg['sky'])
        scene.fog_density = (70, 430)
    except Exception:
        pass
    G['plats'] = generar_pista(n, cfg, rng, root)
    crear_meta(G['plats'][-1], root)
    G['frutas'] = crear_frutas(G['plats'], cfg, rng, root)
    zmax = G['plats'][-1].z
    decorar(cfg, rng, root, zmax)
    crear_aves(cfg, rng, root, zmax)
    G['clf'] = entrenar_ia(n)
    G['t_nivel'] = 0.0
    G['llegadas'] = 0
    G['cuenta_final'] = None


# ------------------------------------------------------------------ RANKING (pandas + CSV)
COLS_RANK = ['fecha', 'jugador', 'animal', 'modo', 'puntos']
NOMBRES_MODO = {1: 'PC vs Usuario', 2: 'PC vs Usuario vs Usuario', 3: 'Usuario vs Usuario'}


def cargar_ranking():
    """Lee el ranking historico (o devuelve una tabla vacia)."""
    try:
        if os.path.exists(ARCHIVO_RANKING):
            df = pd.read_csv(ARCHIVO_RANKING, encoding='utf-8-sig')
            if 'jugador' in df.columns and 'puntos' in df.columns:
                df['jugador'] = df['jugador'].astype(str)
                df['puntos'] = pd.to_numeric(df['puntos'], errors='coerce').fillna(0).astype(int)
                return df
    except Exception as e:
        print('No se pudo leer el ranking:', e)
    return pd.DataFrame(columns=COLS_RANK)


def guardar_ranking(filas):
    """Agrega las partidas nuevas al ranking historico."""
    nuevo = pd.DataFrame(filas, columns=COLS_RANK)
    df = cargar_ranking()
    df = nuevo if df.empty else pd.concat([df, nuevo], ignore_index=True)
    try:
        df.to_csv(ARCHIVO_RANKING, index=False, encoding='utf-8-sig')
    except Exception as e:
        print('No se pudo guardar el ranking:', e)


def tabla_ranking():
    """Una fila por jugador: mejor puntaje y numero de partidas."""
    df = cargar_ranking()
    if df.empty:
        return pd.DataFrame(columns=['jugador', 'mejor', 'partidas'])
    g = df.groupby('jugador').agg(mejor=('puntos', 'max'), partidas=('puntos', 'size')).reset_index()
    return g.sort_values(['mejor', 'partidas'], ascending=[False, True]).reset_index(drop=True)


# ------------------------------------------------------------------ UI (tarjetas, botones, textos)
FONDO = '#16263bf0'
BORDE = '#ffb347'
CLARO = '#fff4dc'
SUAVE = '#a9b8d4'
ORO = '#ffd24a'
FILA = '#243b57'
OSCURO = '#16263b'
MEDALLAS = ['#ffd24a', '#d5dbe6', '#e0985a']


def ui_clear():
    """Borra toda la interfaz dinamica (menus, HUD, etc.)."""
    for e in G['ui']:
        try:
            destroy(e)
        except Exception:
            pass
    G['ui'] = []
    G['hud'] = None
    G['ui_cuenta'] = None
    G['ui_campo'] = None


def ui_rect(x, y, w, h, col, z=1.0):
    e = Entity(parent=camera.ui, model='quad', position=(x, y, z), scale=(w, h), color=C(col))
    transp(e)
    G['ui'].append(e)
    return e


def ui_text(s, x, y, size=1.0, col=CLARO, origin=(0, 0)):
    t = Text(text=tx(s), parent=camera.ui, origin=origin, position=(x, y), scale=size, color=C(col))
    G['ui'].append(t)
    return t


def ui_tarjeta(w, h, cy=0.0):
    ui_rect(0, cy, w + 0.016, h + 0.016, BORDE, 2.2)
    ui_rect(0, cy, w, h, FONDO, 2.0)


def ui_titulo(s, y, size=2.6, col=ORO):
    return ui_text(s, 0, y, size, col)


def ui_linea(y, w=0.8, col=BORDE):
    ui_rect(0, y, w, 0.004, col, 1.0)


def ui_boton(y, tecla, titulo, sub, col):
    ui_rect(0, y, 0.86, 0.095, FILA, 1.2)
    ui_rect(-0.37, y, 0.075, 0.075, col, 1.0)
    ui_text(tecla, -0.37, y, 1.7, OSCURO)
    ui_text(titulo, -0.31, y + 0.018, 1.25, CLARO, (-0.5, 0))
    ui_text(sub, -0.31, y - 0.02, 0.8, SUAVE, (-0.5, 0))


# ------------------------------------------------------------------ PANTALLAS
def mostrar_menu():
    G['estado'] = 'menu'
    ui_clear()
    ui_tarjeta(1.1, 0.92)
    ui_titulo('AVENTURA SALVAJE', 0.36, 3.0)
    ui_text('COLOMBIA 3D', 0, 0.28, 1.8, CLARO)
    ui_linea(0.225)
    ui_text('Plataformas por los ecosistemas de Colombia', 0, 0.185, 0.95, SUAVE)
    ui_boton(0.09, '1', 'PC vs Usuario', 'Tú (Jaguar) contra el PC (Tití cabeciblanco)', '#ffb347')
    ui_boton(-0.025, '2', 'PC vs Usuario vs Usuario', 'Jaguar, Oso de anteojos y el PC', '#7ddc9a')
    ui_boton(-0.14, '3', 'Usuario vs Usuario', 'Jaguar contra Oso de anteojos', '#ff7eb6')
    ui_boton(-0.255, '4', 'Ranking', 'Los mejores exploradores de Colombia', '#7fb2ff')
    ui_text('Jugador 1: W A S D + ESPACIO', 0, -0.345, 0.8, SUAVE)
    ui_text('Jugador 2: Flechas + CTRL / SHIFT derecho', 0, -0.378, 0.8, SUAVE)
    ui_text('Pulsa 1, 2, 3 o 4   ·   ESC para salir', 0, -0.428, 0.9, BORDE)


def nombre_fmt(s):
    """Primera letra de cada palabra en mayuscula."""
    return ' '.join(w[:1].upper() + w[1:] for w in s.split(' '))


def defecto_nombre(paso):
    if paso < len(G['ultimos']) and G['ultimos'][paso]:
        return G['ultimos'][paso]
    return 'Jugador %d' % (paso + 1)


def pedir_nombres(modo):
    G['modo_pend'] = modo
    G['nombres'] = []
    G['nom_buf'] = ''
    G['estado'] = 'nombre'
    pantalla_nombre()


def pantalla_nombre():
    paso = len(G['nombres'])
    modo = G['modo_pend']
    animal = 'Jaguar' if paso == 0 else 'Oso de anteojos'
    colj = '#6ea8ff' if paso == 0 else '#ff7eb6'
    ui_clear()
    ui_tarjeta(0.95, 0.66)
    ui_titulo('¿CÓMO TE LLAMAS?', 0.22, 2.4)
    ui_linea(0.165, 0.6)
    ui_text('Jugador %d   ·   %s' % (paso + 1, animal), 0, 0.115, 1.3, colj)
    ui_rect(0, 0.015, 0.64, 0.115, colj, 1.2)
    ui_rect(0, 0.015, 0.625, 0.1, '#0e1a2b', 1.0)
    G['ui_campo'] = ui_text('', -0.29, 0.015, 1.8, CLARO, (-0.5, 0))
    G['cur_ant'] = None
    refrescar_campo()
    ui_text('Letras y números  ·  máximo %d caracteres' % MAX_NOMBRE, 0, -0.065, 0.8, SUAVE)
    ui_text('Si lo dejas vacío serás "%s"' % defecto_nombre(paso), 0, -0.1, 0.8, SUAVE)
    if paso == 1:
        ui_text('Jugador 1: %s' % G['nombres'][0], 0, -0.16, 0.9, '#6ea8ff')
    ui_text('ENTER para continuar   ·   ESC para volver', 0, -0.26, 0.95, BORDE)


def refrescar_campo():
    if G['ui_campo'] is None:
        return
    cur = '|' if int(G['t'] * 2) % 2 == 0 else ' '
    G['cur_ant'] = cur
    G['ui_campo'].text = tx(nombre_fmt(G['nom_buf']) + cur)


def tecla_nombre(key):
    buf = G['nom_buf']
    if key in ('enter', 'numpad enter'):
        confirmar_nombre()
        return
    if key == 'escape':
        mostrar_menu()
        return
    if key == 'backspace':
        G['nom_buf'] = buf[:-1]
    elif key == 'space':
        if buf and not buf.endswith(' ') and len(buf) < MAX_NOMBRE:
            G['nom_buf'] = buf + ' '
    elif len(key) == 1 and key.isascii() and key.isalnum():
        if len(buf) < MAX_NOMBRE:
            G['nom_buf'] = buf + key.lower()
    refrescar_campo()


def confirmar_nombre():
    paso = len(G['nombres'])
    n = nombre_fmt(G['nom_buf']).strip()
    if not n:
        n = defecto_nombre(paso)
    if n.lower() == 'pc':
        n += ' 1'
    if paso == 1 and n.lower() == G['nombres'][0].lower():
        n += ' 2'
    G['nombres'].append(n)
    G['nom_buf'] = ''
    total = 1 if G['modo_pend'] == 1 else 2
    if len(G['nombres']) >= total:
        G['ultimos'] = list(G['nombres'])
        iniciar_juego(G['modo_pend'], list(G['nombres']))
    else:
        pantalla_nombre()


def mostrar_intro():
    cfg = NIVELES[G['nivel']]
    fa = FAUNA[FAUNA['nivel'] == G['nivel'] + 1]
    ui_clear()
    ui_tarjeta(1.3, 0.86)
    ui_rect(0, 0.345, 0.34, 0.05, '#ff7a59', 1.0)
    ui_text('NIVEL %d  /  %d' % (G['nivel'] + 1, len(NIVELES)), 0, 0.345, 1.0, OSCURO)
    ui_titulo(cfg['nombre'].upper(), 0.255, 2.6)
    ui_text(cfg['sub'], 0, 0.17, 1.05, SUAVE)
    ui_linea(0.125, 1.0)
    ui_text('FAUNA Y FLORA DE ESTE NIVEL', 0, 0.085, 0.95, BORDE)
    for i, r in enumerate(fa.itertuples()):
        y = 0.005 - i * 0.12
        ui_rect(0, y, 1.2, 0.105, FILA, 1.2)
        ui_rect(-0.593, y, 0.014, 0.105, cfg['top'], 1.0)
        ui_text('%s   ·   %s' % (r.animal, r.cientifico), -0.56, y + 0.02, 1.1, CLARO, (-0.5, 0))
        ui_text(r.dato, -0.56, y - 0.024, 0.85, SUAVE, (-0.5, 0))
    G['ui_cuenta'] = ui_text('Prepárate para la aventura...', 0, -0.375, 1.2, BORDE)


# ------------------------------------------------------------------ HUD
def crear_hud():
    ui_clear()
    stats = []
    for i, p in enumerate(G['jug']):
        y = 0.455 - i * 0.085
        ui_rect(-0.55, y, 0.62, 0.075, '#ffffffd0', 1.2)
        ui_rect(-0.857, y, 0.014, 0.075, p.col_marc, 1.0)
        ui_text('%s  ·  %s' % (p.nombre, p.animal['nombre']), -0.83, y + 0.017, 1.0, p.col_txt, (-0.5, 0))
        stats.append(ui_text('', -0.83, y - 0.017, 0.8, '#44546f', (-0.5, 0)))
    ui_rect(0.64, 0.455, 0.44, 0.075, '#ffffffd0', 1.2)
    t_niv = ui_text('', 0.64, 0.472, 0.95, '#2b3a55')
    t_tiempo = ui_text('', 0.64, 0.438, 0.85, '#44546f')
    aviso_r = ui_rect(0, 0.36, 0.78, 0.07, '#ffe066ee', 1.2)
    aviso = ui_text('', 0, 0.36, 1.5, '#b8322a')
    aviso_r.enabled = False
    ui_rect(0, -0.478, 1.25, 0.04, '#16263bb8', 1.2)
    ui_text('R reiniciar nivel   ·   ESC menú   ·   Rosado y lila: plataformas que se mueven', 0, -0.478, 0.8, CLARO)
    G['hud'] = dict(stats=stats, nivel=t_niv, tiempo=t_tiempo, aviso=aviso, aviso_r=aviso_r)


def refrescar_hud():
    h = G['hud']
    if not h:
        return
    for t, p in zip(h['stats'], G['jug']):
        t.text = tx('Frutas %d   ·   Puntos %d' % (p.frutas, p.total + p.frutas) + ('   ·   ¡META!' if p.llego else ''))
    cfg = NIVELES[G['nivel']]
    h['nivel'].text = tx('Nivel %d: %s' % (G['nivel'] + 1, cfg['nombre']))
    h['tiempo'].text = tx('Tiempo %d s' % int(G['t_nivel']))
    if G['cuenta_final'] is not None:
        h['aviso'].text = tx('¡Hay un ganador! Quedan %d s' % max(0, int(G['cuenta_final'])))
        h['aviso_r'].enabled = True
    else:
        h['aviso'].text = ''
        h['aviso_r'].enabled = False


# ------------------------------------------------------------------ FLUJO DEL JUEGO
CONTROLES_J1 = dict(izq=('a',), der=('d',), arr=('w',), abj=('s',), salto=('space',))
CONTROLES_J2 = dict(izq=('left arrow',), der=('right arrow',), arr=('up arrow',), abj=('down arrow',),
                    salto=('right control', 'right shift', 'enter'))
CONTROLES_J1_DOBLE = dict(izq=('a', 'left arrow'), der=('d', 'right arrow'), arr=('w', 'up arrow'),
                          abj=('s', 'down arrow'), salto=('space',))

ANIMALES = dict(
    jaguar=dict(tipo='jaguar', nombre='Jaguar'),
    oso=dict(tipo='oso', nombre='Oso de anteojos'),
    titi=dict(tipo='titi', nombre='Tití cabeciblanco'),
)


def crear_jugadores(modo, nombres):
    for p in G['jug']:
        p.destruir()
    jug = []
    n1 = nombres[0] if len(nombres) > 0 else 'Jugador 1'
    n2 = nombres[1] if len(nombres) > 1 else 'Jugador 2'
    if modo == 1:
        jug.append(Jugador('J1', n1, ANIMALES['jaguar'], False, CONTROLES_J1_DOBLE, '#6ea8ff', '#2f5fd0'))
        jug.append(Jugador('PC', 'PC', ANIMALES['titi'], True, None, '#7ddc9a', '#2d9a58', 0.9))
    elif modo == 2:
        jug.append(Jugador('J1', n1, ANIMALES['jaguar'], False, CONTROLES_J1, '#6ea8ff', '#2f5fd0'))
        jug.append(Jugador('J2', n2, ANIMALES['oso'], False, CONTROLES_J2, '#ff7eb6', '#d03a82'))
        jug.append(Jugador('PC', 'PC', ANIMALES['titi'], True, None, '#7ddc9a', '#2d9a58', 0.9))
    else:
        jug.append(Jugador('J1', n1, ANIMALES['jaguar'], False, CONTROLES_J1, '#6ea8ff', '#2f5fd0'))
        jug.append(Jugador('J2', n2, ANIMALES['oso'], False, CONTROLES_J2, '#ff7eb6', '#d03a82'))
    G['jug'] = jug


def colocar_jugadores():
    n = len(G['jug'])
    for i, p in enumerate(G['jug']):
        x = -3 + 6 * i / max(1, n - 1)
        p.reiniciar(x, 0.0, -2.5)
        p.rot = 0.0
        p.cuerpo.rotation_y = 0


def empezar_nivel():
    cargar_nivel(G['nivel'])
    colocar_jugadores()
    G['estado'] = 'intro'
    G['t_intro'] = 5.0
    mostrar_intro()
    camara(0, duro=True)


def iniciar_juego(modo, nombres):
    G['modo'] = modo
    G['nivel'] = 0
    G['hist'] = []
    G['recien'] = []
    crear_jugadores(modo, nombres)
    empezar_nivel()


def volver_menu():
    for p in G['jug']:
        p.destruir()
    G['jug'] = []
    G['nivel'] = 0
    cargar_nivel(0)
    mostrar_menu()


def leer_teclas(c):
    ix = sum(held_keys[k] for k in c['der']) - sum(held_keys[k] for k in c['izq'])
    iz = sum(held_keys[k] for k in c['arr']) - sum(held_keys[k] for k in c['abj'])
    ix, iz = cl(ix, -1, 1), cl(iz, -1, 1)
    m = math.hypot(ix, iz)
    if m > 1:
        ix, iz = ix / m, iz / m
    saltar = any(held_keys[k] for k in c['salto'])
    return ix, iz, saltar


def ordinal(n):
    return f'{n}.º' if n else '—'


def finalizar_nivel():
    cfg = NIVELES[G['nivel']]
    for p in G['jug']:
        bonus = BONUS.get(p.puesto, 0) if p.llego else 0
        p.pts_nivel = p.frutas + bonus
        p.total += p.pts_nivel
        G['hist'].append(dict(
            fecha=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            modo=G['modo'], nivel=G['nivel'] + 1, ecosistema=cfg['nombre'],
            jugador=p.nombre, animal=p.animal['nombre'], frutas=p.frutas,
            puesto=p.puesto if p.llego else 0, puntos_nivel=p.pts_nivel))
    ultimo = G['nivel'] == len(NIVELES) - 1
    orden = sorted(G['jug'], key=lambda q: -q.pts_nivel)
    ui_clear()
    ui_tarjeta(1.3, 0.78)
    ui_titulo('¡NIVEL %d COMPLETADO!' % (G['nivel'] + 1), 0.31, 2.5)
    ui_text(cfg['nombre'], 0, 0.235, 1.2, SUAVE)
    ui_linea(0.19, 1.0)
    for x, s in ((0.12, 'Frutas'), (0.26, 'Llegada'), (0.39, 'Nivel'), (0.52, 'Total')):
        ui_text(s, x, 0.15, 0.85, BORDE)
    ui_text('Jugador', -0.5, 0.15, 0.85, BORDE, (-0.5, 0))
    for i, p in enumerate(orden):
        y = 0.065 - i * 0.105
        ui_rect(0, y, 1.22, 0.092, FILA, 1.2)
        ui_rect(-0.604, y, 0.014, 0.092, p.col_marc, 1.0)
        ui_text(str(i + 1), -0.56, y, 1.3, MEDALLAS[min(i, 2)])
        ui_text(p.nombre, -0.5, y + 0.017, 1.2, CLARO, (-0.5, 0))
        ui_text(p.animal['nombre'], -0.5, y - 0.02, 0.7, SUAVE, (-0.5, 0))
        ui_text(str(p.frutas), 0.12, y, 1.2)
        ui_text(ordinal(p.puesto if p.llego else 0), 0.26, y, 1.2)
        ui_text('+%d' % p.pts_nivel, 0.39, y, 1.2, '#7ddc9a')
        ui_text(str(p.total), 0.52, y, 1.4, ORO)
    if ultimo:
        G['estado'] = 'fin_nivel_ultimo'
        ui_text('ENTER para ver el resultado final', 0, -0.335, 1.0, BORDE)
    else:
        G['estado'] = 'fin_nivel'
        ui_text('ENTER siguiente nivel   ·   ESC volver al menú', 0, -0.335, 1.0, BORDE)


def mostrar_final():
    df = pd.DataFrame(G['hist'])
    try:
        df.to_csv(ARCHIVO_RESULTADOS, index=False, encoding='utf-8-sig')
    except Exception as e:
        print('No se pudo guardar el CSV:', e)
    rank = (df.groupby(['jugador', 'animal'], as_index=False)['puntos_nivel'].sum()
              .sort_values('puntos_nivel', ascending=False).reset_index(drop=True))
    print('\n===== RANKING FINAL =====')
    print(rank.to_string(index=False))

    # guardar en el ranking historico (solo jugadores humanos)
    humanos = [p for p in G['jug'] if not p.es_bot]
    fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    guardar_ranking([(fecha, p.nombre, p.animal['nombre'], G['modo'], p.total) for p in humanos])
    G['recien'] = [p.nombre for p in humanos]
    hist = tabla_ranking()
    notas = []
    for p in humanos:
        pos = hist.index[hist['jugador'] == p.nombre]
        if len(pos):
            notas.append('%s: puesto %d del ranking histórico' % (p.nombre, int(pos[0]) + 1))

    mejor = rank['puntos_nivel'].max()
    ganadores = rank[rank['puntos_nivel'] == mejor]
    if len(ganadores) > 1:
        titulo = '¡EMPATE!'
    else:
        g = ganadores.iloc[0]
        titulo = '¡GANA %s!' % g['jugador'].upper()

    ui_clear()
    ui_tarjeta(1.2, 0.84)
    ui_titulo(titulo, 0.345, 2.8)
    ui_text('Fin de la aventura por Colombia', 0, 0.265, 1.1, SUAVE)
    ui_linea(0.225, 0.9)
    ui_text('Jugador', -0.46, 0.185, 0.85, BORDE, (-0.5, 0))
    ui_text('Puntos', 0.45, 0.185, 0.85, BORDE)
    cols = {p.nombre: p.col_marc for p in G['jug']}
    for i, r in enumerate(rank.itertuples()):
        y = 0.1 - i * 0.105
        ui_rect(0, y, 1.1, 0.092, FILA, 1.2)
        ui_rect(-0.545, y, 0.014, 0.092, cols.get(r.jugador, '#ffffff'), 1.0)
        ui_rect(-0.49, y, 0.06, 0.06, MEDALLAS[min(i, 2)], 1.0)
        ui_text(str(i + 1), -0.49, y, 1.4, OSCURO)
        ui_text(r.jugador, -0.44, y + 0.017, 1.3, CLARO, (-0.5, 0))
        ui_text(r.animal, -0.44, y - 0.021, 0.75, SUAVE, (-0.5, 0))
        ui_text(str(r.puntos_nivel), 0.45, y, 1.8, ORO)
    ny = 0.1 - len(rank) * 0.105 - 0.01
    for k, s in enumerate(notas):
        ui_text(s, 0, ny - k * 0.04, 0.9, '#7ddc9a')
    ui_text('Resultados guardados en %s' % ARCHIVO_RANKING, 0, -0.33, 0.75, SUAVE)
    ui_text('ENTER para ver el ranking histórico', 0, -0.385, 1.0, BORDE)
    G['estado'] = 'fin_juego'


def mostrar_ranking():
    G['estado'] = 'ranking'
    ui_clear()
    t = tabla_ranking()
    ui_tarjeta(1.2, 0.92)
    ui_titulo('RANKING DE EXPLORADORES', 0.395, 2.3)
    ui_text('Mejor puntaje de cada jugador', 0, 0.335, 1.0, SUAVE)
    ui_linea(0.3, 0.95)
    ui_text('#', -0.52, 0.27, 0.85, BORDE)
    ui_text('Jugador', -0.44, 0.27, 0.85, BORDE, (-0.5, 0))
    ui_text('Mejor', 0.25, 0.27, 0.85, BORDE)
    ui_text('Partidas', 0.45, 0.27, 0.85, BORDE)
    if t.empty:
        ui_text('Aún no hay partidas guardadas.', 0, 0.03, 1.3, CLARO)
        ui_text('¡Termina una aventura y sé el primero!', 0, -0.03, 1.0, SUAVE)
    for i, r in enumerate(t.head(TOP_RANKING).itertuples()):
        y = 0.215 - i * 0.062
        nuevo = r.jugador in G['recien']
        ui_rect(0, y, 1.1, 0.054, '#3a4f33' if nuevo else FILA, 1.2)
        if i < 3:
            ui_rect(-0.52, y, 0.04, 0.04, MEDALLAS[i], 1.0)
            ui_text(str(i + 1), -0.52, y, 1.0, OSCURO)
        else:
            ui_text(str(i + 1), -0.52, y, 1.0, SUAVE)
        ui_text(r.jugador, -0.44, y, 1.15, ORO if nuevo else CLARO, (-0.5, 0))
        ui_text(str(r.mejor), 0.25, y, 1.3, ORO)
        ui_text(str(r.partidas), 0.45, y, 1.0, SUAVE)
    ui_text('ENTER o ESC para volver al menú', 0, -0.425, 0.95, BORDE)


def salir_ranking():
    G['recien'] = []
    if G['jug']:
        volver_menu()
    else:
        mostrar_menu()


def input(key):
    est = G['estado']
    if est == 'menu':
        if key in ('1', '2', '3'):
            pedir_nombres(int(key))
        elif key == '4':
            mostrar_ranking()
        elif key == 'escape':
            application.quit()
    elif est == 'nombre':
        tecla_nombre(key)
    elif est == 'ranking':
        if key in ('enter', 'escape'):
            salir_ranking()
    elif est in ('intro', 'jugando'):
        if key == 'escape':
            volver_menu()
        elif key == 'r':
            empezar_nivel()
    elif est == 'fin_nivel':
        if key == 'enter':
            G['nivel'] += 1
            empezar_nivel()
        elif key == 'escape':
            volver_menu()
    elif est == 'fin_nivel_ultimo':
        if key == 'enter':
            mostrar_final()
    elif est == 'fin_juego':
        if key == 'enter':
            mostrar_ranking()
        elif key == 'escape':
            volver_menu()


# ------------------------------------------------------------------ CAMARA / AMBIENTE
def camara(dt, duro=False):
    ps = G['jug']
    if not ps:
        return
    zl = max(p.z for p in ps)
    zt = min(p.z for p in ps)
    sp = min(zl - zt, 32)
    cx = sum(p.x for p in ps) / len(ps) * 0.6
    cy = sum(p.y for p in ps) / len(ps)
    objetivo = Vec3(cx, cy + 9 + sp * 0.12, (zl + zt) / 2 - 15 - sp * 0.5)
    if duro:
        camera.position = objetivo
    else:
        camera.position = lr(camera.position, objetivo, min(1.0, dt * 3.0))
    camera.rotation = (22, 0, 0)


def camara_menu(dt):
    t = G['t'] * 0.15
    camera.position = Vec3(math.sin(t) * 26, 15, 8 + math.cos(t) * 10)
    camera.look_at(Vec3(0, 2, 30))


def ambiente(dt):
    t = G['t']
    for e, v in G['nubes']:
        e.x += v * dt
        if e.x > 110:
            e.x = -110
    for a in G['aves']:
        ang = t * a['vel'] + a['fase']
        a['e'].position = (a['cx'] + math.cos(ang) * a['rad'],
                           a['cy'] + math.sin(ang * 1.7) * 1.2,
                           a['cz'] + math.sin(ang) * a['rad'] * 0.6)
        dx = -math.sin(ang) * a['rad'] * a['vel']
        dz = math.cos(ang) * a['rad'] * 0.6 * a['vel']
        a['e'].rotation_y = math.degrees(math.atan2(dx, dz))
        fl = math.sin(t * 8 + a['fase']) * 30
        a['alas'][0].rotation_z = fl
        a['alas'][1].rotation_z = -fl
    for b in G['banderas']:
        b.rotation_y = math.sin(t * 3) * 14
    for f in G['frutas']:
        if f.activa:
            f.animar(t)
    if G['sol'] is not None:
        G['sol'].position = camera.position + Vec3(-40, 40, 420)


def actualizar_plataformas(dt):
    for p in G['plats']:
        p.actualizar(G['t_nivel'] if G['estado'] == 'jugando' else G['t'])


def jugar(dt):
    plats = G['plats']
    G['t_nivel'] += dt
    nivel_idx = G['nivel']
    ps = G['jug']
    for p in ps:
        if p.llego:
            ix = iz = 0.0
            saltar = False
            vmax = VEL_HUMANO
        elif p.es_bot:
            ix, iz, saltar = p.ia(dt, plats, G['clf'], nivel_idx)
            vmax = 6.2 + 0.55 * nivel_idx
        else:
            ix, iz, saltar = leer_teclas(p.teclas)
            vmax = VEL_HUMANO
        p.fisica(dt, ix, iz, saltar, vmax, plats)
    # empujones suaves entre jugadores
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, b = ps[i], ps[j]
            dx, dz = b.x - a.x, b.z - a.z
            d = math.hypot(dx, dz)
            if 0.001 < d < 1.1 and abs(a.y - b.y) < 1.2:
                push = (1.1 - d) / 2
                a.x -= dx / d * push
                a.z -= dz / d * push
                b.x += dx / d * push
                b.z += dz / d * push
    # frutas
    for f in G['frutas']:
        if not f.activa:
            continue
        for p in ps:
            if abs(p.x - f.x) < 1.2 and abs(p.z - f.z) < 1.2 and abs((p.y + 0.8) - f.ent.y) < 1.8:
                f.activa = False
                f.ent.enabled = False
                p.frutas += 1
                break
    # llegada a la meta
    meta = plats[-1]
    for p in ps:
        if not p.llego and p.suelo is meta:
            p.llego = True
            G['llegadas'] += 1
            p.puesto = G['llegadas']
            if G['cuenta_final'] is None:
                G['cuenta_final'] = 10.0
    # no dejar a nadie demasiado atras
    zl = max(p.z for p in ps)
    lider = max(ps, key=lambda q: q.z)
    for p in ps:
        if not p.llego and zl - p.z > 36:
            ref = plats[max(0, lider.idx_actual - 2)]
            p.ultima_segura = (ref.bx, ref.bt + 0.6, ref.bz)
            p.reaparecer()
    for p in ps:
        p.sincronizar()
        p.animar(dt, plats)
    # fin del nivel
    if G['cuenta_final'] is not None:
        G['cuenta_final'] -= dt
    if all(p.llego for p in ps) or (G['cuenta_final'] is not None and G['cuenta_final'] <= 0) \
            or G['t_nivel'] > LIMITE_NIVEL:
        finalizar_nivel()
        return
    camara(dt)
    G['hud_t'] -= dt
    if G['hud_t'] <= 0:
        refrescar_hud()
        G['hud_t'] = 0.12


def update():
    dt = min(time.dt, 1 / 30)
    G['t'] += dt
    est = G['estado']
    ambiente(dt)
    if est in ('menu', 'nombre', 'ranking'):
        actualizar_plataformas(dt)
        camara_menu(dt)
        if est == 'nombre':
            cur = '|' if int(G['t'] * 2) % 2 == 0 else ' '
            if cur != G['cur_ant']:
                refrescar_campo()
    elif est == 'intro':
        actualizar_plataformas(dt)
        for p in G['jug']:
            p.fisica(dt, 0, 0, False, VEL_HUMANO, G['plats'])
            p.sincronizar()
            p.animar(dt, G['plats'])
        camara(dt)
        G['t_intro'] -= dt
        if G['t_intro'] <= 0:
            G['estado'] = 'jugando'
            G['t_nivel'] = 0.0
            crear_hud()
            refrescar_hud()
        elif G['t_intro'] < 3.2 and G['ui_cuenta'] is not None:
            G['ui_cuenta'].text = tx('¡Prepárate!  Empieza en %d' % (int(G['t_intro']) + 1))
    elif est == 'jugando':
        actualizar_plataformas(dt)
        jugar(dt)
    else:  # pantallas de resultados: el mundo sigue vivo
        actualizar_plataformas(dt)
        for p in G['jug']:
            p.fisica(dt, 0, 0, False, VEL_HUMANO, G['plats'])
            p.sincronizar()
            p.animar(dt, G['plats'])
        camara(dt)


# ------------------------------------------------------------------ INICIO
if __name__ == '__main__':
    try:
        cargar_nivel(0)
        mostrar_menu()
        app.run()
    except SystemExit:
        raise
    except BaseException:
        err = traceback.format_exc()
        print('\n[ERROR] El juego se detuvo:\n')
        print(err)
        try:
            with open('error_aventura_salvaje.txt', 'w', encoding='utf-8') as f:
                f.write(err)
            print('(El error tambien se guardo en error_aventura_salvaje.txt)')
        except Exception:
            pass
        print('\nPulsa ENTER para cerrar...')
        _pausa()
        sys.exit(1)
