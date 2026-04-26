@echo off
setlocal

cd /d "%~dp0"

set "PY_EXE=C:\Users\gaele\AppData\Local\Programs\Python\Python313\python.exe"

echo Reiniciando backend en puerto 8000...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
    echo Cerrando proceso PID %%P que usa el puerto 8000...
    taskkill /PID %%P /F >nul 2>&1
)

echo Iniciando backend en http://127.0.0.1:8000 ...
if exist "%PY_EXE%" (
    start "Backend Chat IA" cmd /k "cd /d ""%~dp0"" & ""%PY_EXE%"" -m uvicorn app:app --host 127.0.0.1 --port 8000"
) else (
    start "Backend Chat IA" cmd /k "cd /d ""%~dp0"" & py -3 -m uvicorn app:app --host 127.0.0.1 --port 8000"
)

echo Abriendo interfaz web...
start "" "%~dp0index.html"

echo.
echo Listo. Si el navegador abre antes de que el backend termine de iniciar,
echo refresca la pagina en unos segundos.
exit /b 0
