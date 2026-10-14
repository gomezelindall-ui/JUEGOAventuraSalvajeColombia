# 🦜 Aventura Salvaje Colombia (2D)

Juego de plataformas en 2D hecho con **Python y Pygame**. Elige un animal de la fauna colombiana, sube por las plataformas, recoge frutas y llega primero a la bandera a lo largo de **5 niveles** inspirados en las regiones de Colombia.

La PC está controlada por un **árbol de decisión (scikit-learn)** entrenado con datos sintéticos generados con **pandas y NumPy**.

## Características

- 3 modos de juego: *Usuario vs PC*, *PC vs Usuario vs Usuario* y *Usuario vs Usuario*.
- 3 dificultades contra la PC: **Fácil**, **Medio** y **Difícil**.
- Cada jugador escribe su nombre.
- **Ranking** de partidas ganadas y perdidas (se guarda en `ranking_colombia.csv`).
- No se pasa de nivel hasta ganarle a la PC.
- 6 animales: jaguar, oso de anteojos, perezoso, chigüiro, cóndor andino y rana dorada, con datos curiosos.
- 5 niveles: Selva Amazónica, Páramo Andino, Costa Caribe, Llanos Orientales y Nevados Andinos.

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
python aventura_salvaje_2d.py
```

## Controles

| Modo | Jugador 1 | Jugador 2 |
|------|-----------|-----------|
| Usuario vs PC | `A`/`D` o `←`/`→` para moverse, `W`, `↑` o `ESPACIO` para saltar | — |
| Dos jugadores | `A`/`D` mover, `W` saltar | `←`/`→` mover, `↑` saltar |

- Menús: `↑` `↓` y `ENTER` (o teclas `1`, `2`, `3`).
- `R` en el menú: ver ranking.
- `ESC`: volver al menú.

## Puntuación

- Fruta: **+1** punto
- Bandera: **+5** puntos

## Dificultades

| Nivel | Velocidad PC | Dudas de la PC | Tiempo por nivel |
|-------|--------------|----------------|------------------|
| Fácil | 3.0 | Frecuentes | 90 s |
| Medio | 3.8 | Normales | 80 s |
| Difícil | 4.6 | Casi nunca | 70 s |

## Estructura del proyecto

```
├── aventura_salvaje_2d.py   # código del juego
├── requirements.txt         # dependencias
├── README.md
├── LICENSE
└── .gitignore
```

## Licencia

Distribuido bajo la licencia MIT. Ver el archivo `LICENSE`.
