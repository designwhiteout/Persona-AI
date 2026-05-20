@echo off
echo Building PersonaAI.exe...
pip install pyinstaller -q
pyinstaller persona_ai.spec --clean --noconfirm
echo.
echo Done! Find PersonaAI.exe in dist\
pause
