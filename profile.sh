#!/bin/bash
# Usage: ./profile.sh course2/vec_add.cu

if [ -z "$1" ]; then
    echo "Usage: ./profile.sh <path/to/file.cu>"
    echo "Example: ./profile.sh course2/vec_add.cu"
    exit 1
fi

SRC="$1"
DIR=$(dirname "$SRC")
NAME=$(basename "$SRC" .cu)
OUTDIR="build/$DIR"
EXE="$OUTDIR/$NAME.exe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$OUTDIR/${NAME}_${TIMESTAMP}"

mkdir -p "$OUTDIR"

ROOT=$(pwd -W)

# Compile with lineinfo, then profile with Nsight Compute
cat > _profile_tmp.bat << ENDOFBAT
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "$ROOT"
nvcc -lineinfo "$SRC" -o "$EXE"
if errorlevel 1 exit /b 1
echo --- Profiling with Nsight Compute ---
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1\target\windows-desktop-win7-x64\ncu.exe" --set full --page details -o "$REPORT" "$EXE"
ENDOFBAT

cmd //c "$ROOT\\_profile_tmp.bat"
RESULT=$?
rm -f _profile_tmp.bat

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "Report saved to: $REPORT.ncu-rep"
    echo "Open with: Nsight Compute GUI -> File -> Open"
else
    echo "Failed!"
    exit 1
fi
