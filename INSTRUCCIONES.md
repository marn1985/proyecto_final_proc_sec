# Instalación de la app Whisper en un equipo nuevo

Esta guía instala la app de transcripción y traducción con Whisper (modo Asistente y modo Explorador) desde cero. Está escrita para Windows; al final hay una nota para macOS y Linux.

## Requisitos previos

- **Python 3.11 o 3.12** (recomendado). La app también funcionó con Python 3.14, pero las versiones 3.11 y 3.12 son las de mayor compatibilidad con PyTorch y Whisper. Descárgalo de [python.org](https://www.python.org/downloads/) y, durante la instalación, marca la casilla **"Add python.exe to PATH"**.
- **Conexión a internet** la primera vez, para instalar las librerías y descargar el modelo `base` (unos 140 MB).
- **Unos 3 GB libres en disco**, principalmente por PyTorch.

No hace falta instalar ffmpeg aparte: la librería `imageio-ffmpeg` ya trae uno.

## Paso 1. Preparar la carpeta del proyecto

Crea una carpeta, por ejemplo `D:\Universidad\whisper_app`, y copia en ella estos dos archivos:

```
whisper_app/
├── app_whisper_streamlit.py
└── requirements.txt
```

Abre una terminal (PowerShell o la terminal de VS Code) dentro de esa carpeta:

```powershell
cd D:\Universidad\whisper_app
```

## Paso 2. Crear un entorno virtual

El entorno virtual mantiene las librerías del proyecto separadas del resto del equipo.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell muestra un error de "ejecución de scripts deshabilitada", ejecuta una sola vez el siguiente comando y vuelve a activar el entorno:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sabrás que el entorno está activo porque la línea de la terminal empieza con `(.venv)`.

## Paso 3. Instalar las dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La instalación puede tardar varios minutos por el tamaño de PyTorch.

## Paso 4. Ejecutar la app

```powershell
python -m streamlit run app_whisper_streamlit.py
```

El navegador se abrirá en `http://localhost:8501`. La primera ejecución tarda un poco más porque descarga el modelo Whisper; las siguientes cargan de inmediato.

Para detener la app, presiona `Ctrl + C` en la terminal.

## Uso diario

Cada vez que quieras volver a abrir la app en ese equipo:

```powershell
cd D:\Universidad\whisper_app
.\.venv\Scripts\Activate.ps1
python -m streamlit run app_whisper_streamlit.py
```

## Problemas frecuentes

| Mensaje de error | Causa | Solución |
|---|---|---|
| `streamlit no se reconoce como nombre de un cmdlet` | El ejecutable no está en el PATH | Usa siempre `python -m streamlit run ...` |
| `can't open file '...\streamlit'` | Faltó el `-m` en el comando | Escribe `python -m streamlit`, no `python streamlit` |
| `FileNotFoundError: [WinError 2]` al cargar el audio | No se encuentra ffmpeg | Ejecuta `python -m pip install imageio-ffmpeg` |
| No aparece el clip 📎 en el chat | Streamlit anterior a 1.43 | Ejecuta `python -m pip install -U streamlit` |
| El micrófono no graba | El navegador no tiene permiso | Acepta el permiso del micrófono y abre la app desde `localhost` |
| Falla la instalación de `torch` o `numba` | Versión de Python sin soporte | Instala Python 3.12 y repite desde el Paso 2 |

## Opcional: usar tarjeta gráfica NVIDIA

Por defecto se instala PyTorch para CPU, que es suficiente para el modelo `base`. Si el equipo tiene una GPU NVIDIA y quieres más velocidad, instala la versión de PyTorch con CUDA siguiendo el selector oficial en [pytorch.org/get-started](https://pytorch.org/get-started/locally/). La app detecta la GPU automáticamente.

## Nota para macOS y Linux

Los pasos son los mismos, con dos cambios en los comandos:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app_whisper_streamlit.py
```
