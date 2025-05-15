@echo off
echo Starting Hamster Monitor Main Server...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

:: Activate Conda environment
call conda activate hamster-monitor-env
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'hamster-monitor-env'
    echo Please make sure the environment exists
    pause
    exit /b 1
)
echo Conda environment 'hamster-monitor-env' activated

:: Check if required files exist
if not exist "main.py" (
    echo Error: main.py not found
    pause
    exit /b 1
)

if not exist "best.pt" (
    echo Warning: best.pt model file not found
    echo The activity detection may not work properly
)

:: Start the server
echo Starting server on http://localhost:8081
python main.py

:: If the server crashes, pause to see the error
if errorlevel 1 (
    echo.
    echo Server crashed! Press any key to exit...
    pause
) 