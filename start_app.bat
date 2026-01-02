@echo off
call .venv\Scripts\activate

rem Set environment variables to suppress warnings
set PYTHONWARNINGS=ignore::UserWarning:pydantic._internal._fields
set PYTHONWARNINGS=%PYTHONWARNINGS%,ignore::DeprecationWarning:pkg_resources
set PYTHONPATH=%CD%\src;%PYTHONPATH%

cd chainlit-app
chainlit run app.py --watch
