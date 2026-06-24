#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Stopping services..."

  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi

  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi

  echo "Stopped."
}

trap cleanup SIGINT SIGTERM EXIT

cd "$ROOT_DIR"

if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

export PYTHONPATH="$ROOT_DIR"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import importlib.util
import subprocess
import sys

required = [
    "flask",
    "flask_cors",
    "pandas",
    "numpy",
    "sklearn",
    "sentence_transformers",
    "matplotlib",
]

missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]

if missing:
    print("Installing backend dependencies...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt"
    ])
else:
    print("Backend dependencies already installed.")
PY

if [ ! -f "data/movies_metadata.csv" ]; then
  echo "Using bundled sample dataset. Add data/movies_metadata.csv for full Kaggle/TMDB recommendations."
fi

echo "Starting backend..."
python -m app.api &
BACKEND_PID=$!

echo "Waiting for backend..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:5001/health >/dev/null; then
    echo "Backend is ready."
    break
  fi

  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend failed to start."
    exit 1
  fi

  sleep 2
done

echo "Installing frontend dependencies..."
cd "$ROOT_DIR/frontend"

npm config set registry https://registry.npmjs.org/

if [ ! -d "node_modules" ] || [ ! -f "node_modules/.bin/vite" ]; then
  npm install || {
    echo "npm install failed. Cleaning and retrying..."
    rm -rf node_modules package-lock.json
    npm cache verify
    npm install
  }
else
  echo "Frontend dependencies already installed."
fi

echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!

echo "Waiting for frontend..."
for i in {1..30}; do
  if curl -s http://localhost:5173 >/dev/null; then
    echo "Frontend is ready."
    break
  fi

  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "Frontend failed to start."
    exit 1
  fi

  sleep 1
done

echo ""
echo "Application started successfully"
echo "Backend:  http://127.0.0.1:5001"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

wait