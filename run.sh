#!/bin/bash
# Usage: ./run.sh course1/hello_world.cu

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <path/to/file.cu>"
    echo "Example: ./run.sh course1/hello_world.cu"
    exit 1
fi

SRC="$1"
DIR=$(dirname "$SRC")
NAME=$(basename "$SRC" .cu)
OUTDIR="build/$DIR"
EXE="$OUTDIR/$NAME.exe"

mkdir -p "$OUTDIR"

# Get Windows absolute path of project root
ROOT=$(pwd -W)

# Write a temporary bat file to avoid quoting issues
cat > _build_tmp.bat << ENDOFBAT
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "$ROOT"
nvcc -lineinfo "$SRC" -o "$EXE"
ENDOFBAT

cmd //c "$ROOT\\_build_tmp.bat"
RESULT=$?
rm -f _build_tmp.bat

if [ $RESULT -eq 0 ]; then
    echo "--- Running $EXE ---"
    "./$EXE"
else
    echo "Compilation failed!"
    exit 1
fi
