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
    1. PC vs Usuario              -> eliges tu animal (Jaguar, Oso o Titi) y compites contra el PC
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
    M = silenciar / activar el sonido (musica de fondo distinta en cada nivel)

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
import wave
from pathlib import Path
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

DIR_JUEGO = os.path.dirname(os.path.abspath(__file__))
try:
    os.chdir(DIR_JUEGO)    # los CSV y los sonidos se guardan junto al juego
except Exception:
    pass

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


# ------------------------------------------------------------------ SONIDO
# Todo el audio se genera por codigo (numpy + wave) la primera vez que juegas:
# no hay que descargar nada. Los archivos .wav quedan junto al juego.
SR = 22050
SND_ON = True
SND = {}
MUS = {}
MUS_ACT = [None]

MUSICA = {
    'menu': dict(bpm=100, raiz=60, esc=[0, 2, 4, 7, 9], timbre='tri', semilla=11, bajo=True, perc='suave', pad=True),
    'n0': dict(bpm=104, raiz=57, esc=[0, 3, 5, 7, 10], timbre='flauta', semilla=21, bajo=True, perc='tambor', pad=False),
    'n1': dict(bpm=80, raiz=62, esc=[0, 2, 4, 7, 9], timbre='tri', semilla=22, bajo=True, perc='ninguna', pad=True),
    'n2': dict(bpm=64, raiz=64, esc=[0, 2, 3, 7, 8], timbre='flauta', semilla=23, bajo=False, perc='ninguna', pad=True),
    'n3': dict(bpm=116, raiz=60, esc=[0, 2, 4, 5, 7, 9, 11], timbre='cuadrada', semilla=24, bajo=True, perc='cumbia', pad=False),
    'n4': dict(bpm=132, raiz=55, esc=[0, 2, 4, 5, 7, 9, 10], timbre='arpa', semilla=25, bajo=True, perc='joropo', pad=False),
}


def _hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def _tono(f, dur, timbre='sin', vol=0.4, att=0.01, dec=4.0):
    n = max(16, int(SR * dur))
    t = np.arange(n) / SR
    fase = 2 * np.pi * f * t
    if timbre == 'tri':
        w = 2 / np.pi * np.arcsin(np.sin(fase))
    elif timbre == 'cuadrada':
        w = np.sign(np.sin(fase)) * 0.3 + np.sin(fase) * 0.45
    elif timbre == 'flauta':
        vib = np.sin(2 * np.pi * 5.0 * t) * 0.003 * np.minimum(1.0, t * 3)
        fv = 2 * np.pi * f * (t + vib)
        w = np.sin(fv) + 0.25 * np.sin(2 * fv) + 0.07 * np.sin(3 * fv)
    elif timbre == 'arpa':
        w = np.sin(fase) + 0.5 * np.sin(2 * fase) * np.exp(-6 * t) + 0.2 * np.sin(3 * fase) * np.exp(-9 * t)
    else:
        w = np.sin(fase)
    env = np.minimum(1.0, t / max(att, 1e-4)) * np.exp(-dec * t)
    fin = np.clip((dur - t) / 0.03, 0.0, 1.0)
    return w * env * fin * vol


def _ruido(n, semilla):
    return np.random.RandomState(semilla).uniform(-1, 1, n)


def _golpe(tipo, vol=0.5):
    if tipo == 'bombo':
        n = int(SR * 0.2)
        t = np.arange(n) / SR
        f = 48 + 110 * np.exp(-t * 22)
        w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12)
    elif tipo == 'caja':
        n = int(SR * 0.12)
        t = np.arange(n) / SR
        w = _ruido(n, 5) * np.exp(-t * 30) * 0.8 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4
    else:  # maraca
        n = int(SR * 0.05)
        t = np.arange(n) / SR
        w = np.diff(_ruido(n, 9), prepend=0.0) * np.exp(-t * 70) * 0.6
    return w * vol


def _poner(buf, y, inicio):
    i = int(inicio * SR)
    if i >= len(buf) or i < 0:
        return
    fin = min(len(buf), i + len(y))
    buf[i:fin] += y[:fin - i]


def _pista(cfg, compases=8):
    """Compone una melodia de 8 compases (frase A-B-A-B) con bajo, acordes y percusion."""
    rng = random.Random(cfg['semilla'])
    spb = 60.0 / cfg['bpm']
    beats = compases * 4
    largo = int(beats * spb * SR)
    buf = np.zeros(largo + int(SR * 1.2))
    esc, raiz, perc = cfg['esc'], cfg['raiz'], cfg['perc']
    n_esc = len(esc)
    dec = {'flauta': 1.6, 'arpa': 5.0, 'tri': 3.0, 'cuadrada': 4.0}[cfg['timbre']]
    if perc == 'joropo':
        durs = [0.5, 0.5, 0.25, 0.25, 0.5, 1.0]
    elif cfg['timbre'] == 'flauta':
        durs = [1.0, 2.0, 1.5, 1.0, 2.0]
    else:
        durs = [0.5, 1.0, 1.0, 0.5, 1.5, 2.0]

    def midi(g):
        return raiz + 12 * (g // n_esc) + esc[g % n_esc]

    def frase(total):
        notas, pos, g = [], 0.0, n_esc
        while pos < total - 1e-6:
            d = min(rng.choice(durs), total - pos)
            if rng.random() > 0.12:
                notas.append((pos, g, d))
            g = int(cl(g + rng.choice([-2, -1, -1, 0, 1, 1, 2]), n_esc - 2, n_esc * 2 + 1))
            pos += d
        return notas

    fa, fb = frase(8), frase(8)
    for k, fr in enumerate([fa, fb, fa, fb]):
        for (p, g, d) in fr:
            y = _tono(_hz(midi(g)), d * spb * 0.95, cfg['timbre'], 0.30, 0.01, dec)
            _poner(buf, y, (k * 8 + p) * spb)

    prog = [0, 2, 3, 1] if n_esc == 5 else [0, 4, 5, 3]
    if cfg['bajo']:
        patron = {'cumbia': [(0, 0.7), (1.5, 0.4), (2, 0.7), (3.5, 0.4)],
                  'joropo': [(0, 0.45), (1, 0.45), (2, 0.45), (3, 0.45)]}.get(perc, [(0, 1.5), (2, 1.5)])
        for c in range(compases):
            m = raiz - 12 + esc[prog[c % 4] % n_esc]
            for b, d in patron:
                _poner(buf, _tono(_hz(m), d * spb, 'tri', 0.32, 0.01, 3.0), (c * 4 + b) * spb)
    if cfg['pad']:
        for c in range(compases):
            m = raiz + esc[prog[c % 4] % n_esc]
            for off in (0, 7, 12):
                _poner(buf, _tono(_hz(m + off), 4 * spb, 'sin', 0.06, 0.5, 0.3), c * 4 * spb)
    if perc != 'ninguna':
        golpes = {'tambor': [0, 2], 'cumbia': [0, 2], 'joropo': [0, 1.5, 3], 'suave': [0]}[perc]
        paso = {'tambor': 0.5, 'cumbia': 0.5, 'joropo': 0.25, 'suave': 1.0}[perc]
        for c in range(compases):
            b0 = c * 4
            for b in golpes:
                _poner(buf, _golpe('bombo', 0.3 if perc == 'suave' else 0.5), (b0 + b) * spb)
            k = 0.0
            while k < 4:
                _poner(buf, _golpe('maraca', 0.28), (b0 + k) * spb)
                k += paso
            if perc == 'cumbia':
                for b in (1, 3):
                    _poner(buf, _golpe('caja', 0.3), (b0 + b) * spb)

    cola = buf[largo:].copy()          # la cola se mezcla al inicio para que el bucle no corte
    buf = buf[:largo]
    buf[:len(cola)] += cola[:len(buf)]
    return buf / (np.max(np.abs(buf)) * 1.15 + 1e-9)


def _barrido(f0, f1, dur, vol=0.4, tipo='sin', dec=3.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = f0 + (f1 - f0) * (t / dur)
    fase = 2 * np.pi * np.cumsum(f) / SR
    w = np.sin(fase) if tipo == 'sin' else np.sign(np.sin(fase)) * 0.5
    env = np.minimum(1.0, t / 0.005) * np.exp(-dec * t) * np.clip((dur - t) / 0.02, 0.0, 1.0)
    return w * env * vol


def _seq(notas, timbre='tri', vol=0.4, dec=6.0):
    """notas = [(nota_midi, inicio_seg, duracion_seg), ...]"""
    total = max(i + d for _, i, d in notas) + 0.1
    buf = np.zeros(int(total * SR))
    for m, i, d in notas:
        _poner(buf, _tono(_hz(m), d, timbre, vol, 0.005, dec), i)
    return buf


def _efectos():
    return {
        'salto': _barrido(280, 640, 0.16, 0.35),
        'aterriza': _barrido(150, 60, 0.09, 0.45, 'sin', 10.0),
        'fruta': _seq([(84, 0, 0.07), (91, 0.07, 0.18)], 'tri', 0.4, 5.0),
        'caida': _barrido(520, 90, 0.45, 0.3, 'cuadrada', 2.0),
        'meta': _seq([(72, 0, 0.12), (76, 0.12, 0.12), (79, 0.24, 0.12), (84, 0.36, 0.45)], 'tri', 0.45, 3.0),
        'nivel': _seq([(67, 0, 0.14), (72, 0.14, 0.14), (76, 0.28, 0.14), (79, 0.42, 0.14),
                       (84, 0.56, 0.4), (79, 0.95, 0.12), (84, 1.07, 0.7)], 'arpa', 0.5, 2.5),
        'fin': _seq([(72, 0, 0.2), (76, 0.2, 0.2), (79, 0.4, 0.2), (84, 0.6, 0.35), (79, 1.0, 0.2),
                     (84, 1.2, 0.2), (88, 1.4, 0.2), (91, 1.6, 0.9)], 'arpa', 0.5, 2.0),
        'menu': _seq([(76, 0, 0.06)], 'tri', 0.3, 12.0),
        'ok': _seq([(79, 0, 0.06), (84, 0.06, 0.12)], 'tri', 0.35, 8.0),
        'cuenta': _seq([(72, 0, 0.14)], 'tri', 0.4, 6.0),
        'ya': _seq([(84, 0, 0.4)], 'tri', 0.45, 3.0),
        'alerta': _seq([(88, 0, 0.08), (88, 0.14, 0.08), (88, 0.28, 0.14)], 'cuadrada', 0.25, 6.0),
    }


def _guardar_wav(ruta, y):
    datos = (np.clip(y, -1.0, 1.0) * 32767 * 0.85).astype('<i2')
    with wave.open(ruta, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(datos.tobytes())


def generar_sonidos():
    """Crea los .wav la primera vez (tarda unos segundos)."""
    for nombre, y in _efectos().items():
        ruta = 'snd_%s.wav' % nombre
        if not os.path.exists(ruta):
            _guardar_wav(ruta, y)
    for nombre, cfg in MUSICA.items():
        ruta = 'mus_%s.wav' % nombre
        if not os.path.exists(ruta):
            _guardar_wav(ruta, _pista(cfg))


def iniciar_sonido():
    try:
        generar_sonidos()
    except Exception as e:
        print('No se pudo generar el sonido:', e)
        return
    volumen = dict(salto=0.5, aterriza=0.35, fruta=0.6, caida=0.5, meta=0.7, nivel=0.7,
                   fin=0.8, menu=0.45, ok=0.5, cuenta=0.5, ya=0.6, alerta=0.45)
    for n, v in volumen.items():
        try:
            SND[n] = Audio('snd_%s.wav' % n, autoplay=False, loop=False, volume=v)
        except Exception as e:
            print('Sonido no disponible (%s): %s' % (n, e))
    for n in MUSICA:
        try:
            MUS[n] = Audio('mus_%s.wav' % n, autoplay=False, loop=True, volume=0.32)
        except Exception as e:
            print('Musica no disponible (%s): %s' % (n, e))


def sfx(nombre):
    """Efecto de sonido (salto, fruta, meta...)."""
    if not SND_ON:
        return
    a = SND.get(nombre)
    if a is None:
        return
    try:
        a.clip.stop()
        a.clip.play()
    except Exception:
        pass


def _parar_musica():
    for a in MUS.values():
        try:
            a.clip.stop()
        except Exception:
            pass


def musica(nombre):
    """Cambia la musica de fondo ('menu', 'n0'..'n4')."""
    if MUS_ACT[0] == nombre and SND_ON:
        return
    _parar_musica()
    MUS_ACT[0] = nombre
    a = MUS.get(nombre)
    if SND_ON and a is not None:
        try:
            a.clip.setLoop(True)
            a.clip.play()
        except Exception:
            pass


def alternar_sonido():
    """Tecla M: silencia o activa todo el sonido."""
    global SND_ON
    SND_ON = not SND_ON
    if SND_ON:
        actual = MUS_ACT[0]
        MUS_ACT[0] = None
        musica(actual)
    else:
        _parar_musica()


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
         sky='#9fe3c0', piso='#2a9d8f', top='#7ed957', side='#8a5a3a',
         tronco='#7a5232', hojas=['#2f9e4f', '#46b85a', '#1f8a47'], colina='#3fae6a',
         fruta='#ffb347', deco='selva', ave=('#ff4d4d', '#3d8bff', 1.0),
         sol='#fff3a0', sol_esc=60, sol_off=(-90, 18, 340), nube='#e6fff0ee',
         niebla=(25, 230), mh=(30, 52), fauna_plat='rana'),
    dict(nombre='Valle del Cocora', sub='Palmas de cera entre la neblina del Eje Cafetero',
         sky='#d4e8f7', piso='#4fb35a', top='#d2f07a', side='#6c9a47',
         tronco='#e9dcb4', hojas=['#4fbf6a', '#74d37e'], colina='#86d46f',
         fruta='#ff7f7f', deco='cera', ave=('#ffd84d', '#9be58f', 0.9),
         sol='#fffbe0', sol_esc=55, sol_off=(70, 18, 340), nube='#ffffffee',
         niebla=(18, 160), mh=(24, 44), fauna_plat=None),
    dict(nombre='Páramo andino', sub='Frailejones y cóndores sobre las nubes',
         sky='#c9b6f5', piso='#7f9fd6', top='#e0ead0', side='#7a84a6',
         tronco='#b8a878', hojas=['#e8f2b8', '#cfe49a'], colina='#eef0ff',
         fruta='#ffd36e', deco='frailejon', ave=('#6d6f8c', '#a8aed0', 1.8),
         sol='#ffd9ec', sol_esc=40, sol_off=(80, 22, 340), nube='#ffd9f2ee',
         niebla=(35, 300), mh=(55, 95), fauna_plat=None),
    dict(nombre='Costa Caribe', sub='Playas, palmas de coco y flamencos',
         sky='#6fd8ff', piso='#13b4d8', top='#ffe9a6', side='#e4b86c',
         tronco='#c8a06a', hojas=['#2fb866', '#5ad080'], colina='#ffd9a0',
         fruta='#ffe066', deco='coco', ave=('#ff8fb4', '#ffc2d6', 1.0),
         sol='#fff27a', sol_esc=70, sol_off=(60, 20, 340), nube='#ffffffee',
         niebla=(120, 600), mh=(3, 9), fauna_plat=None),
    dict(nombre='Llanos Orientales', sub='Sabana infinita al atardecer',
         sky='#ff9a68', piso='#d9a24a', top='#9ad45a', side='#8a5a2a',
         tronco='#6e4a2c', hojas=['#3f7f35', '#4e9140'], colina='#c7603a',
         fruta='#ff9ec7', deco='saman', ave=('#ffffff', '#e8f0ff', 1.1),
         sol='#ffe58a', sol_esc=110, sol_off=(-40, 10, 340), nube='#ffb98aee',
         niebla=(90, 700), mh=(6, 14), fauna_plat='chiguiro'),
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

try:
    application.asset_folder = Path(DIR_JUEGO)
except Exception:
    pass
iniciar_sonido()

G = dict(
    estado='menu', modo=0, nivel=0, jug=[], plats=[], frutas=[], root=None,
    aves=[], nubes=[], banderas=[], sol=None, t=0.0, t_nivel=0.0, t_intro=0.0,
    llegadas=0, cuenta_final=None, hist=[], hud_t=0.0, clf=None, rng=random.Random(7),
    # interfaz
    ui=[], hud=None, ui_cuenta=None, ui_campo=None, cur_ant=None,
    # nombres y ranking
    modo_pend=0, nombres=[], nom_buf='', ultimos=[], recien=[],
    sel=0, prev=None, cuenta_ult=-1, elegido='jaguar',
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
def _pata(cu, px, pz, ancho, alto, col, col_pata, garra='#fffaf0', garras=True):
    """Pata articulada: el pivote esta arriba para poder balancearla al caminar."""
    r = Entity(parent=cu, position=(px, alto, pz))
    caja(r, (0, -alto / 2, 0), (ancho, alto, ancho), C(col))
    caja(r, (0, -alto + 0.07, ancho * 0.22), (ancho * 1.2, 0.17, ancho * 1.5), C(col_pata), 'sphere')
    if garras:
        for k in (-1, 0, 1):
            caja(r, (k * ancho * 0.32, -alto + 0.05, ancho * 0.92), (ancho * 0.15, 0.07, ancho * 0.26), C(garra))
    return r


def _ojo(cu, x, y, z, r, iris, lado=1, esclera='#ffffff'):
    caja(cu, (x, y, z), (r, r, r * 0.6), C(esclera), 'sphere')
    caja(cu, (x, y, z + r * 0.22), (r * 0.62, r * 0.62, r * 0.4), C(iris), 'sphere')
    caja(cu, (x, y, z + r * 0.38), (r * 0.34, r * 0.34, r * 0.2), C('#1d1d2b'), 'sphere')
    caja(cu, (x + r * 0.12 * lado, y + r * 0.14, z + r * 0.46), (r * 0.14, r * 0.14, r * 0.1), C('#ffffff'), 'sphere')


def _jaguar(cu):
    pel, claro, osc, rosa = '#f4b95a', '#fff1d2', '#7c4a22', '#ff9aa8'
    caja(cu, (0, 0.88, -0.02), (0.98, 0.78, 1.65), C(pel), 'sphere')        # cuerpo
    caja(cu, (0, 0.66, 0.0), (0.82, 0.36, 1.35), C(claro), 'sphere')        # vientre
    caja(cu, (0, 1.03, 0.58), (0.98, 0.86, 0.8), C(pel), 'sphere')          # hombros
    caja(cu, (0, 1.05, 0.88), (0.62, 0.6, 0.5), C(pel), 'sphere')           # cuello
    # rosetas del lomo
    for (mx, mz) in [(-0.25, 0.35), (0.22, 0.2), (-0.18, -0.1), (0.25, -0.3), (-0.22, -0.45), (0.05, -0.6),
                     (0.0, 0.1), (-0.3, 0.0), (0.3, -0.05), (0.1, -0.28), (-0.05, 0.5), (0.15, 0.55)]:
        t = 1 - (mx / 0.49) ** 2 - ((mz + 0.02) / 0.82) ** 2
        if t > 0.05:
            my = 0.88 + 0.39 * math.sqrt(t)
            caja(cu, (mx, my - 0.005, mz), (0.22, 0.07, 0.22), C(osc), 'sphere')
            caja(cu, (mx, my + 0.014, mz), (0.11, 0.07, 0.11), C(pel), 'sphere')
    # manchas de los costados
    for sz in (-0.55, -0.25, 0.05, 0.35):
        for sy in (0.98, 0.78):
            for lado in (-1, 1):
                t = max(0.05, 1 - ((sz + 0.02) / 0.82) ** 2 - ((sy - 0.88) / 0.39) ** 2)
                caja(cu, (lado * (0.49 * math.sqrt(t) - 0.01), sy, sz), (0.06, 0.2, 0.2), C(osc), 'sphere')
    # cabeza
    caja(cu, (0, 1.2, 1.12), (0.86, 0.78, 0.8), C(pel), 'sphere')
    for lado in (-1, 1):
        caja(cu, (0.3 * lado, 1.08, 1.22), (0.36, 0.3, 0.3), C(claro), 'sphere')      # mejillas
    caja(cu, (0, 1.06, 1.5), (0.46, 0.32, 0.34), C(claro), 'sphere')                # hocico
    caja(cu, (0, 1.15, 1.66), (0.2, 0.13, 0.1), C(rosa), 'sphere')                  # nariz
    caja(cu, (0, 1.03, 1.64), (0.02, 0.1, 0.02), C(osc))
    caja(cu, (0, 0.99, 1.62), (0.26, 0.015, 0.02), C(osc))
    for (hx, hz) in [(-0.14, 1.0), (0.14, 1.0), (0.0, 1.12)]:
        caja(cu, (hx, 1.56, hz), (0.1, 0.05, 0.1), C(osc), 'sphere')
    for lado in (-1, 1):
        _ojo(cu, 0.25 * lado, 1.3, 1.45, 0.2, '#e8a020', lado)
        caja(cu, (0.25 * lado, 1.46, 1.43), (0.2, 0.04, 0.05), C(osc), rotation_z=-12 * lado)
        caja(cu, (0.34 * lado, 1.56, 0.96), (0.3, 0.3, 0.18), C(osc), 'sphere')        # orejas
        caja(cu, (0.34 * lado, 1.56, 1.02), (0.24, 0.24, 0.14), C(pel), 'sphere')
        caja(cu, (0.34 * lado, 1.56, 1.07), (0.14, 0.14, 0.08), C(rosa), 'sphere')
        for dy, rz in ((0.05, 10), (-0.02, 0), (-0.09, -10)):
            caja(cu, (lado * 0.42, 1.05 + dy, 1.5), (0.4, 0.014, 0.014), C('#3a2a22'), rotation_z=rz * lado)
    # cola con anillos
    cola = Entity(parent=cu, position=(0, 1.0, -0.78))
    for i in range(6):
        caja(cola, (0, 0.03 * i * i, -0.15 - 0.2 * i), (0.22, 0.22, 0.26), C(osc if i % 2 else pel), 'sphere')
    caja(cola, (0, 0.03 * 36, -1.45), (0.24, 0.24, 0.26), C(osc), 'sphere')
    cu.cola = cola
    patas = []
    for (px, pz) in [(-0.3, 0.55), (0.3, 0.55), (-0.3, -0.55), (0.3, -0.55)]:
        patas.append(_pata(cu, px, pz, 0.3, 0.55, pel, '#ffe0a0'))
    return patas


def _oso(cu):
    pel, claro, hoc, osc = '#4a3d45', '#f6e7cc', '#ecd8b6', '#241b20'
    caja(cu, (0, 0.98, -0.05), (1.22, 1.05, 1.65), C(pel), 'sphere')            # cuerpo
    caja(cu, (0, 1.32, 0.35), (1.0, 0.7, 0.85), C(pel), 'sphere')               # joroba
    caja(cu, (0, 0.98, 0.7), (0.78, 0.8, 0.5), C(claro), 'sphere')              # babero crema
    caja(cu, (0, 1.3, 1.02), (1.0, 0.95, 0.92), C(pel), 'sphere')               # cabeza
    caja(cu, (0, 1.17, 1.45), (0.6, 0.48, 0.46), C(hoc), 'sphere')              # hocico
    caja(cu, (0, 1.28, 1.67), (0.24, 0.17, 0.13), C(osc), 'sphere')             # nariz
    caja(cu, (0, 1.08, 1.66), (0.02, 0.1, 0.02), C(osc))
    caja(cu, (0, 1.03, 1.62), (0.22, 0.015, 0.02), C(osc))
    for lado in (-1, 1):
        caja(cu, (0.29 * lado, 1.46, 1.38), (0.38, 0.38, 0.2), C(claro), 'sphere')   # "anteojos"
        _ojo(cu, 0.29 * lado, 1.46, 1.45, 0.15, '#3b2a22', lado, '#f4ead6')
        caja(cu, (0.4 * lado, 1.3, 1.33), (0.09, 0.28, 0.09), C(claro), rotation_z=-14 * lado)
        caja(cu, (0.42 * lado, 1.74, 0.9), (0.34, 0.34, 0.26), C(pel), 'sphere')     # orejas
        caja(cu, (0.42 * lado, 1.74, 1.0), (0.2, 0.2, 0.1), C('#8a6e68'), 'sphere')
    cola = Entity(parent=cu, position=(0, 1.0, -0.85))
    caja(cola, (0, 0, -0.05), (0.24, 0.24, 0.26), C(pel), 'sphere')
    cu.cola = cola
    patas = []
    for (px, pz) in [(-0.4, 0.5), (0.4, 0.5), (-0.4, -0.5), (0.4, -0.5)]:
        patas.append(_pata(cu, px, pz, 0.46, 0.56, pel, osc, claro))
    return patas


def _titi(cu):
    pel, vientre, cara, blanco = '#7b5a46', '#e0c8ae', '#5d4856', '#ffffff'
    caja(cu, (0, 0.88, 0), (0.62, 0.7, 0.9), C(pel), 'sphere')                  # cuerpo
    caja(cu, (0, 0.82, 0.12), (0.48, 0.52, 0.72), C(vientre), 'sphere')         # barriga
    caja(cu, (0, 1.34, 0.42), (0.7, 0.66, 0.64), C(cara), 'sphere')             # cabeza
    caja(cu, (0, 1.24, 0.72), (0.36, 0.28, 0.2), C('#8e7585'), 'sphere')        # hocico
    caja(cu, (0, 1.29, 0.82), (0.1, 0.06, 0.05), C('#2a1f2a'), 'sphere')
    caja(cu, (0, 1.17, 0.81), (0.14, 0.015, 0.02), C('#2a1f2a'))
    for k in range(-4, 5):                                                        # cresta blanca
        caja(cu, (0.07 * k, 1.78, 0.34), (0.075, 0.6, 0.075), C(blanco), rotation_z=k * 9)
        caja(cu, (0.07 * k, 1.7, 0.22), (0.075, 0.7, 0.075), C('#f1f1f6'), rotation_z=k * 7)
    for lado in (-1, 1):
        _ojo(cu, 0.17 * lado, 1.4, 0.7, 0.19, '#2a1f2a', lado)
        caja(cu, (0.37 * lado, 1.5, 0.3), (0.12, 0.55, 0.16), C(blanco), rotation_z=-14 * lado)
        caja(cu, (0.3 * lado, 1.18, 0.25), (0.1, 0.35, 0.12), C('#f1f1f6'), rotation_z=-8 * lado)
        caja(cu, (0.36 * lado, 1.34, 0.4), (0.16, 0.2, 0.1), C('#8e7585'), 'sphere')    # orejas
    cola = Entity(parent=cu, position=(0, 0.95, -0.5))
    for (a, b, c) in [(0, 0, -0.15), (0, 0.04, -0.4), (0, 0.16, -0.62), (0, 0.4, -0.78),
                      (0, 0.7, -0.8), (0, 0.95, -0.68), (0, 1.1, -0.5)]:
        caja(cola, (a, b, c), (0.15, 0.15, 0.15), C('#6a4b3a'), 'sphere')
    cu.cola = cola
    patas = []
    for (px, pz) in [(-0.22, 0.32), (0.22, 0.32), (-0.22, -0.3), (0.22, -0.3)]:
        patas.append(_pata(cu, px, pz, 0.18, 0.58, pel, '#4b382d', garras=False))
    return patas


def construir_animal(tipo, cu):
    """Arma el animal con cubos y esferas. Devuelve la lista de patas."""
    if tipo == 'jaguar':
        return _jaguar(cu)
    if tipo == 'oso':
        return _oso(cu)
    return _titi(cu)


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
        if not s.es_bot:
            sfx('caida')

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
            if not s.es_bot:
                sfx('salto')
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
            if s.suelo is None and s.vy < -9 and not s.es_bot:
                sfx('aterriza')
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
        cola = getattr(s.cuerpo, 'cola', None)
        if cola is not None:
            mov = v > 0.8
            cola.rotation_y = math.sin(G['t'] * (9 if mov else 3)) * (24 if mov else 10)
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


def pico_nevado(root, x, z, h, w):
    """Montaña escalonada con la cima nevada."""
    for k in range(6):
        f = 1 - k / 6.0
        col = '#ffffff' if k >= 4 else ('#9aa3c8' if k >= 2 else '#7a84a8')
        caja(root, (x, FLOOR_Y + h * k / 6.0 + h / 12.0, z), (w * f, h / 6.0 + 0.2, w * f), C(col), rotation_y=45)


def velero(root, x, z, rng):
    e = Entity(parent=root, position=(x, FLOOR_Y + 0.3, z), rotation_y=rng.uniform(0, 360), scale=3)
    caja(e, (0, 0.4, 0), (1.6, 0.8, 4.2), C('#c8553d'))
    caja(e, (0, 2.5, 0), (0.15, 4.2, 0.15), C('#ffffff'))
    caja(e, (0, 2.8, -0.2), (0.08, 3.2, 2.2), C('#ffffff'))


def vaca(root, x, z, rng):
    e = Entity(parent=root, position=(x, FLOOR_Y, z), rotation_y=rng.uniform(0, 360), scale=3.5)
    caja(e, (0, 1.0, 0), (0.9, 0.8, 1.8), C('#fff4e6'))
    caja(e, (0.25, 1.1, 0.2), (0.4, 0.4, 0.5), C('#3b3236'))
    caja(e, (0, 1.25, 1.15), (0.6, 0.6, 0.65), C('#fff4e6'))
    for sx in (-1, 1):
        caja(e, (0.3 * sx, 0.3, 0.6), (0.2, 0.6, 0.2), C('#e8dccb'))
        caja(e, (0.3 * sx, 0.3, -0.6), (0.2, 0.6, 0.2), C('#e8dccb'))


def _extras_selva(root, rng, zmax):
    for _ in range(46):                                  # lianas colgantes
        x = rng.choice((-1, 1)) * rng.uniform(14, 42)
        z = rng.uniform(-20, zmax + 40)
        h = rng.uniform(14, 34)
        caja(root, (x, 34 - h / 2, z), (0.16, h, 0.16), C('#2e7d3c'))
        for k in range(3):
            caja(root, (x + rng.uniform(-0.3, 0.3), 34 - h * (k + 1) / 4, z), (0.7, 0.5, 0.7), C('#58c26a'), 'sphere')
    for _ in range(4):                                   # cascadas
        x = rng.choice((-1, 1)) * rng.uniform(70, 110)
        z = rng.uniform(0, zmax + 60)
        caja(root, (x, FLOOR_Y + 22, z), (7, 44, 3), C('#e2fbff'))
        caja(root, (x, FLOOR_Y + 1, z), (14, 3, 10), C('#ffffff'), 'sphere')
    for _ in range(70):                                  # hojas flotando en el río
        caja(root, (rng.uniform(-60, 60), FLOOR_Y + 0.08, rng.uniform(-20, zmax + 60)),
             (rng.uniform(2, 4.5), 0.12, rng.uniform(2, 4.5)), C('#5fd06a'), 'sphere')


def _extras_cera(root, rng, zmax):
    verdes = ['#86d46f', '#6fc060', '#9ae07a']
    for _ in range(18):                                  # lomas de pasto
        lado = rng.choice((-1, 1))
        caja(root, (lado * rng.uniform(32, 75), FLOOR_Y + 2, rng.uniform(-40, zmax + 120)),
             (rng.uniform(40, 80), rng.uniform(14, 30), rng.uniform(40, 80)), C(rng.choice(verdes)), 'sphere')
    for _ in range(7):                                   # cordillera azulada
        lado = rng.choice((-1, 1))
        caja(root, (lado * rng.uniform(150, 220), FLOOR_Y + 15, rng.uniform(0, zmax + 200)), (80, 70, 80), C('#9db7d8'), 'sphere')
    for _ in range(14):                                  # bancos de neblina
        e = caja(root, (rng.choice((-1, 1)) * rng.uniform(30, 70), FLOOR_Y + rng.uniform(3, 14), rng.uniform(-30, zmax + 100)),
                 (rng.uniform(18, 30), rng.uniform(3, 6), rng.uniform(20, 40)), C('#ffffffaa'), 'sphere')
        transp(e)


def _extras_paramo(root, rng, zmax):
    for _ in range(16):                                  # picos nevados
        pico_nevado(root, rng.choice((-1, 1)) * rng.uniform(40, 130), rng.uniform(-30, zmax + 140),
                    rng.uniform(40, 90), rng.uniform(30, 55))
    for _ in range(5):                                   # lagunas
        caja(root, (rng.choice((-1, 1)) * rng.uniform(25, 70), FLOOR_Y + 0.06, rng.uniform(0, zmax + 60)),
             (rng.uniform(25, 45), 0.1, rng.uniform(15, 30)), C('#5aa6ea'), 'sphere')
    for _ in range(60):                                  # rocas
        caja(root, (rng.choice((-1, 1)) * rng.uniform(14, 60), FLOOR_Y + 0.5, rng.uniform(-20, zmax + 60)),
             (rng.uniform(1, 3), rng.uniform(1, 2.5), rng.uniform(1, 3)),
             C(rng.choice(['#9aa3b8', '#b5bccf', '#7f889f'])), rotation=(0, rng.uniform(0, 90), 0))


def _extras_caribe(root, rng, zmax):
    for _ in range(5):                                   # islas con palmeras
        x = rng.choice((-1, 1)) * rng.uniform(40, 90)
        z = rng.uniform(0, zmax + 80)
        caja(root, (x, FLOOR_Y, z), (rng.uniform(22, 36), 6, rng.uniform(18, 28)), C('#ffe3a0'), 'sphere')
        for _k in range(3):
            palma_coco(root, x + rng.uniform(-6, 6), z + rng.uniform(-5, 5), 14, NIVELES[3], rng)
    for _ in range(9):                                   # veleros
        velero(root, rng.choice((-1, 1)) * rng.uniform(25, 120), rng.uniform(0, zmax + 80), rng)
    for _ in range(80):                                  # espuma de las olas
        caja(root, (rng.uniform(-90, 90), FLOOR_Y + 0.08, rng.uniform(-20, zmax + 80)),
             (rng.uniform(4, 9), 0.1, 0.5), C('#e8fbff'))


def _extras_llanos(root, rng, zmax):
    for _ in range(14):                                  # ganado
        vaca(root, rng.choice((-1, 1)) * rng.uniform(30, 75), rng.uniform(0, zmax + 60), rng)
    for _ in range(130):                                 # pasto seco
        caja(root, (rng.uniform(-80, 80), FLOOR_Y + 1.0, rng.uniform(-20, zmax + 80)),
             (0.15, rng.uniform(1.4, 2.6), 0.15), C(rng.choice(['#f2cf6a', '#d9a441', '#c98f33'])),
             rotation_z=rng.uniform(-12, 12))
    for lado in (-1, 1):                                 # cercas de la hacienda
        z = -10.0
        while z < zmax + 40:
            caja(root, (lado * 26, FLOOR_Y + 3.5, z), (0.6, 7, 0.6), C('#6e4a2c'))
            caja(root, (lado * 26, FLOOR_Y + 5.5, z + 4), (0.35, 0.35, 8.2), C('#8a5f38'))
            caja(root, (lado * 26, FLOOR_Y + 3.0, z + 4), (0.35, 0.35, 8.2), C('#8a5f38'))
            z += 8


def decorar(cfg, rng, root, zmax):
    n = int(max(34, zmax / 3.2))
    if cfg['deco'] == 'selva':
        n = int(n * 1.5)
    f = dict(selva=arbol_selva, cera=palma_cera, frailejon=frailejon_gigante,
             coco=palma_coco, saman=saman)[cfg['deco']]
    for _ in range(n):
        x = rng.choice((-1, 1)) * rng.uniform(15, 44)
        z = rng.uniform(-25, zmax + 45)
        f(root, x, z, rng.uniform(16, 30), cfg, rng)
    # suelo / agua lejano
    caja(root, (0, FLOOR_Y - 0.5, zmax / 2), (420, 1, zmax + 420), C(cfg['piso']))
    # colinas y montañas (la altura cambia en cada ecosistema)
    mh = cfg['mh']
    for _ in range(16):
        lado = rng.choice((-1, 1))
        caja(root, (lado * rng.uniform(70, 140), FLOOR_Y + mh[0] * 0.2, rng.uniform(-40, zmax + 120)),
             (rng.uniform(40, 80), rng.uniform(mh[0], mh[1]), rng.uniform(40, 80)), C(cfg['colina']), 'sphere')
    # nubes (cada nivel tiene su color)
    G['nubes'] = []
    for _ in range(16):
        e = Entity(parent=root, position=(rng.uniform(-90, 90), rng.uniform(24, 50), rng.uniform(-30, zmax + 140)))
        for k in range(3):
            caja(e, (k * 5 - 5, rng.uniform(-0.5, 0.8), rng.uniform(-1, 1)),
                 (rng.uniform(8, 13), rng.uniform(2.5, 4), rng.uniform(5, 8)), C(cfg['nube']), 'sphere')
        G['nubes'].append((e, rng.uniform(0.6, 1.6)))
    # sol (sin sombreado y sin niebla para que siempre se vea)
    G['sol'] = caja(root, (0, 0, 0), cfg['sol_esc'], C(cfg['sol']), 'sphere', shader=None)
    try:
        G['sol'].setFogOff()
    except Exception:
        pass
    dict(selva=_extras_selva, cera=_extras_cera, frailejon=_extras_paramo,
         coco=_extras_caribe, saman=_extras_llanos)[cfg['deco']](root, rng, zmax)


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
        scene.fog_density = cfg['niebla']
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
    musica('menu')
    ui_clear()
    ui_tarjeta(1.1, 0.92)
    ui_titulo('AVENTURA SALVAJE', 0.36, 3.0)
    ui_text('COLOMBIA 3D', 0, 0.28, 1.8, CLARO)
    ui_linea(0.225)
    ui_text('Plataformas por los ecosistemas de Colombia', 0, 0.185, 0.95, SUAVE)
    ui_boton(0.09, '1', 'PC vs Usuario', 'Elige tu animal y compite contra el PC', '#ffb347')
    ui_boton(-0.025, '2', 'PC vs Usuario vs Usuario', 'Jaguar, Oso de anteojos y el PC', '#7ddc9a')
    ui_boton(-0.14, '3', 'Usuario vs Usuario', 'Jaguar contra Oso de anteojos', '#ff7eb6')
    ui_boton(-0.255, '4', 'Ranking', 'Los mejores exploradores de Colombia', '#7fb2ff')
    ui_text('Jugador 1: W A S D + ESPACIO', 0, -0.345, 0.8, SUAVE)
    ui_text('Jugador 2: Flechas + CTRL / SHIFT derecho', 0, -0.378, 0.8, SUAVE)
    ui_text('Pulsa 1, 2, 3 o 4   ·   M sonido   ·   ESC para salir', 0, -0.428, 0.9, BORDE)


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
    sfx('ok')
    if len(G['nombres']) >= total:
        G['ultimos'] = list(G['nombres'])
        if G['modo_pend'] == 1:
            iniciar_seleccion()
        else:
            iniciar_juego(G['modo_pend'], list(G['nombres']))
    else:
        pantalla_nombre()


OPCIONES = ['jaguar', 'oso', 'titi']
INFO_ANIMAL = dict(
    jaguar=('Jaguar', 'Panthera onca', 'Fuerte y feroz: el felino más grande de América'),
    oso=('Oso de anteojos', 'Tremarctos ornatus', 'El único oso de Suramérica, valiente y trepador'),
    titi=('Tití cabeciblanco', 'Saguinus oedipus', 'Pequeño y ágil, con su cresta blanca inconfundible'),
)
COLOR_ANIMAL = dict(jaguar='#ffb347', oso='#7ddc9a', titi='#ff7eb6')


def iniciar_seleccion():
    G['estado'] = 'elegir'
    G['sel'] = 0
    crear_preview()
    pantalla_seleccion()


def quitar_preview():
    p = G.get('prev')
    if p:
        try:
            destroy(p['root'])
        except Exception:
            pass
    G['prev'] = None


def crear_preview():
    """Plataforma flotante muy arriba del mundo donde gira el animal elegido."""
    quitar_preview()
    root = Entity(position=(0, 80, 0))
    caja(root, (0, -0.2, 0), (5.2, 0.35, 5.2), C('#ffe9b0'), 'sphere')
    caja(root, (0, -0.1, 0), (4.4, 0.35, 4.4), C('#ffd27a'), 'sphere')
    G['prev'] = dict(root=root, ent=None)
    cambiar_preview()


def cambiar_preview():
    p = G.get('prev')
    if not p:
        return
    if p['ent'] is not None:
        destroy(p['ent'])
    tipo = OPCIONES[G['sel']]
    ent = Entity(parent=p['root'], rotation_y=150)
    cu = Entity(parent=ent)
    construir_animal(tipo, cu)
    cu.scale = 1.7 * ANIMALES[tipo]['esc']
    p['ent'] = ent


def pantalla_seleccion():
    tipo = OPCIONES[G['sel']]
    nom, cien, dato = INFO_ANIMAL[tipo]
    col = COLOR_ANIMAL[tipo]
    pc = INFO_ANIMAL['titi' if tipo != 'titi' else 'jaguar'][0]
    ui_clear()
    ui_rect(0, 0.42, 1.0, 0.11, FONDO, 2.0)
    ui_titulo('ELIGE TU PERSONAJE', 0.42, 2.3)
    ui_tarjeta(1.2, 0.36, -0.27)
    ui_text(nom.upper(), 0, -0.14, 1.9, col)
    ui_text(cien, 0, -0.185, 0.9, SUAVE)
    ui_text(dato, 0, -0.228, 1.0, CLARO)
    for i, t in enumerate(OPCIONES):
        x = (i - 1) * 0.3
        sel = (i == G['sel'])
        ui_rect(x, -0.30, 0.26, 0.05, COLOR_ANIMAL[t] if sel else FILA, 1.2)
        ui_text('%d  %s' % (i + 1, INFO_ANIMAL[t][0].split(' ')[0]), x, -0.30, 0.85, OSCURO if sel else SUAVE)
    ui_text('El PC jugará con: %s' % pc, 0, -0.355, 0.85, '#7ddc9a')
    ui_text('A / D o flechas: cambiar   ·   ENTER: jugar   ·   ESC: volver', 0, -0.415, 0.85, BORDE)


def tecla_elegir(key):
    if key in ('left arrow', 'a'):
        G['sel'] = (G['sel'] - 1) % 3
    elif key in ('right arrow', 'd'):
        G['sel'] = (G['sel'] + 1) % 3
    elif key in ('1', '2', '3'):
        G['sel'] = int(key) - 1
    elif key in ('enter', 'numpad enter'):
        sfx('ok')
        elegido = OPCIONES[G['sel']]
        quitar_preview()
        iniciar_juego(1, list(G['nombres']), elegido)
        return
    elif key == 'escape':
        quitar_preview()
        mostrar_menu()
        return
    else:
        return
    sfx('menu')
    cambiar_preview()
    pantalla_seleccion()


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
    ui_text('R reiniciar  ·  ESC menú  ·  M sonido  ·  Rosado y lila: plataformas móviles', 0, -0.478, 0.8, CLARO)
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
    jaguar=dict(tipo='jaguar', nombre='Jaguar', esc=1.0),
    oso=dict(tipo='oso', nombre='Oso de anteojos', esc=1.0),
    titi=dict(tipo='titi', nombre='Tití cabeciblanco', esc=0.9),
)


def crear_jugadores(modo, nombres, elegido='jaguar'):
    for p in G['jug']:
        p.destruir()
    jug = []
    n1 = nombres[0] if len(nombres) > 0 else 'Jugador 1'
    n2 = nombres[1] if len(nombres) > 1 else 'Jugador 2'
    A = ANIMALES
    titi = A['titi']
    if modo == 1:
        pj = A[elegido]
        bot = titi if elegido != 'titi' else A['jaguar']      # el PC toma otro animal
        jug.append(Jugador('J1', n1, pj, False, CONTROLES_J1_DOBLE, '#6ea8ff', '#2f5fd0', pj['esc']))
        jug.append(Jugador('PC', 'PC', bot, True, None, '#7ddc9a', '#2d9a58', bot['esc']))
    elif modo == 2:
        jug.append(Jugador('J1', n1, A['jaguar'], False, CONTROLES_J1, '#6ea8ff', '#2f5fd0', A['jaguar']['esc']))
        jug.append(Jugador('J2', n2, A['oso'], False, CONTROLES_J2, '#ff7eb6', '#d03a82', A['oso']['esc']))
        jug.append(Jugador('PC', 'PC', titi, True, None, '#7ddc9a', '#2d9a58', titi['esc']))
    else:
        jug.append(Jugador('J1', n1, A['jaguar'], False, CONTROLES_J1, '#6ea8ff', '#2f5fd0', A['jaguar']['esc']))
        jug.append(Jugador('J2', n2, A['oso'], False, CONTROLES_J2, '#ff7eb6', '#d03a82', A['oso']['esc']))
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
    G['cuenta_ult'] = -1
    musica('n%d' % G['nivel'])
    mostrar_intro()
    camara(0, duro=True)


def iniciar_juego(modo, nombres, elegido='jaguar'):
    G['modo'] = modo
    G['elegido'] = elegido
    G['nivel'] = 0
    G['hist'] = []
    G['recien'] = []
    crear_jugadores(modo, nombres, elegido)
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
    sfx('nivel')
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
    sfx('fin')
    musica('menu')
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
    if key == 'm' and est != 'nombre':
        alternar_sonido()
        return
    if est == 'menu':
        if key in ('1', '2', '3'):
            sfx('ok')
            pedir_nombres(int(key))
        elif key == '4':
            sfx('menu')
            mostrar_ranking()
        elif key == 'escape':
            application.quit()
    elif est == 'nombre':
        tecla_nombre(key)
    elif est == 'elegir':
        tecla_elegir(key)
    elif est == 'ranking':
        if key in ('enter', 'escape'):
            sfx('menu')
            salir_ranking()
    elif est in ('intro', 'jugando'):
        if key == 'escape':
            volver_menu()
        elif key == 'r':
            empezar_nivel()
    elif est == 'fin_nivel':
        if key == 'enter':
            sfx('ok')
            G['nivel'] += 1
            empezar_nivel()
        elif key == 'escape':
            volver_menu()
    elif est == 'fin_nivel_ultimo':
        if key == 'enter':
            sfx('ok')
            mostrar_final()
    elif est == 'fin_juego':
        if key == 'enter':
            sfx('menu')
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
    objetivo = Vec3(cx, cy + 7.8 + sp * 0.12, (zl + zt) / 2 - 15 - sp * 0.5)
    if duro:
        camera.position = objetivo
    else:
        camera.position = lr(camera.position, objetivo, min(1.0, dt * 3.0))
    camera.rotation = (18, 0, 0)


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
        G['sol'].position = camera.position + Vec3(*NIVELES[G['nivel']]['sol_off'])


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
                if not p.es_bot:
                    sfx('fruta')
                break
    # llegada a la meta
    meta = plats[-1]
    for p in ps:
        if not p.llego and p.suelo is meta:
            p.llego = True
            G['llegadas'] += 1
            p.puesto = G['llegadas']
            if not p.es_bot:
                sfx('meta')
            if G['cuenta_final'] is None:
                G['cuenta_final'] = 10.0
                if p.es_bot:
                    sfx('alerta')
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
            sfx('ya')
            G['estado'] = 'jugando'
            G['t_nivel'] = 0.0
            crear_hud()
            refrescar_hud()
        elif G['t_intro'] < 3.2 and G['ui_cuenta'] is not None:
            n_cuenta = int(G['t_intro']) + 1
            G['ui_cuenta'].text = tx('¡Prepárate!  Empieza en %d' % n_cuenta)
            if n_cuenta != G['cuenta_ult']:
                G['cuenta_ult'] = n_cuenta
                sfx('cuenta')
    elif est == 'elegir':
        actualizar_plataformas(dt)
        camera.position = Vec3(0, 82.4, -9.8)
        camera.rotation = (12, 0, 0)
        pv = G['prev']
        if pv and pv['ent'] is not None:
            pv['ent'].rotation_y += 45 * dt
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
