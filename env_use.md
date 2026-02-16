# Requisitos para el funcionamiento del sistema
# Instalación de UV
```
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Posiblemente te pida que ejecutas esto: Set-ExecutionPolicy RemoteSigned -scope CurrentUser

# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

# Instalar todas las dependencias
uv pip install -r requirements.txt
```

# Ejecutar el programa
```
uv run main.py
```