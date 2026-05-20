#!/bin/bash
set -e
echo "Building Persona AI V2..."
pip install pyinstaller -q
pyinstaller persona_ai.spec --clean --noconfirm
echo "Done! Find 'Persona AI V2' in dist/"
