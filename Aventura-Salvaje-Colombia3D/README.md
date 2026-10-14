# 🐆 Aventura Salvaje Colombia 3D

Videojuego 3D de plataformas hecho con **Python y Ursina**. Corre con animales y paisajes de Colombia, recoge frutas y llega primero a la bandera de Colombia a lo largo de **5 ecosistemas**.

El "cerebro" de la PC es un **árbol de decisión (scikit-learn)** que se entrena al iniciar cada nivel. El historial y el ranking se manejan con **pandas** y se guardan en CSV.

## Características

- 3 modos de juego más una pantalla de ranking.
- Cada jugador escribe su nombre al empezar.
- Ranking histórico guardado en `ranking_aventura_salvaje.csv`.
- 5 niveles: Selva Amazónica, Valle del Cocora, Páramo andino, Costa Caribe y Llanos Orientales.
- Fauna colombiana: Jaguar, Oso de anteojos, Titi cabeciblanco y más.

## Requisitos

- Python 3.9 o superior

## Instalación

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

# (opcional) entorno virtual
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

## Ejecutar

```bash
python aventura_salvaje_colombia_3d.py
```

## Modos de juego

1. **PC vs Usuario**: tú (Jaguar) contra la PC (Titi cabeciblanco).
2. **PC vs Usuario vs Usuario**: Jaguar + Oso de anteojos + PC.
3. **Usuario vs Usuario**: Jaguar contra Oso de anteojos.
4. **Ranking**: mejores exploradores (historial guardado).

## Controles

| Jugador | Controles |
|---------|-----------|
| Jugador 1 | `W` `A` `S` `D` + `ESPACIO` |
| Jugador 2 | Flechas + `CTRL` derecho / `SHIFT` derecho / `ENTER` |

En el modo 1 puedes usar `WASD` o las flechas.

Otras teclas: `R` reiniciar nivel · `ESC` volver al menú · `ENTER` continuar.

## Cómo se gana

Es una carrera de plataformas hasta la bandera. Se suman puntos por:

- Frutas recogidas: **1** punto cada una.
- Orden de llegada: **5, 3 y 1** puntos.

Gana quien tenga más puntos al terminar los 5 niveles.

## Archivos que genera el juego

- `resultados_aventura_salvaje.csv`: detalle de cada partida.
- `ranking_aventura_salvaje.csv`: ranking histórico.
- `error_aventura_salvaje.txt`: solo si el juego se detiene por un error.

## Solución de problemas

- **Falta una librería**: instálala con `python -m pip install ursina` (o la que indique el mensaje).
- **El juego se cierra con un error**: revisa `error_aventura_salvaje.txt`.

## Estructura del proyecto

```
├── aventura_salvaje_colombia_3d.py   # código del juego
├── requirements.txt                  # dependencias
├── README.md
├── LICENSE
└── .gitignore
```

## Licencia

Distribuido bajo la licencia MIT. Ver el archivo `LICENSE`.
