#!/bin/bash
set -e
echo "Building PersonaAI..."
pip install pyinstaller -q
pyinstaller persona_ai.spec --clean --noconfirm
echo "Done! Find PersonaAI in dist/"
