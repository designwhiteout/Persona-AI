@echo off
echo Building Persona AI V2.exe...
pip install pyinstaller -q
pyinstaller persona_ai.spec --clean --noconfirm
echo.
echo Done! Find "Persona AI V2.exe" in dist\
pause
