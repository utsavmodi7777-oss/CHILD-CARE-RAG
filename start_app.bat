@echo off
cd /d "%~dp0"
set "VENV_PYTHON=%~dp0.venv_win\Scripts\python.exe"

rem Set environment variables to suppress warnings
set PYTHONWARNINGS=ignore::UserWarning:pydantic._internal._fields
set PYTHONWARNINGS=%PYTHONWARNINGS%,ignore::DeprecationWarning:pkg_resources
set PYTHONPATH=%CD%\src;%PYTHONPATH%

cd chainlit-app
"%VENV_PYTHON%" -m chainlit run app.py --watch
