# Inicia el servidor de desarrollo con el entorno virtual env
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

& "$PSScriptRoot\env\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload