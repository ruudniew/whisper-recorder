#!/bin/bash
# WhisperLive Transcription - All-in-one script
#
# Usage:
#   ./run.sh           - Start the app
#   ./run.sh --setup   - Initial setup (install dependencies)
#   ./run.sh --help    - Show help

# Show help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "WhisperLive Transcription"
    echo ""
    echo "Usage: ./run.sh [option]"
    echo ""
    echo "Options:"
    echo "  (no option)     Start the app"
    echo "  --setup         Initial setup (create venv and install dependencies)"
    echo "  --help, -h      Show this help message"
    echo ""
    exit 0
fi

# Setup mode
if [ "$1" = "--setup" ]; then
    echo "🎙️  Setting up WhisperLive Transcription..."
    
    # Check if Python 3.8+ is installed
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed."
        echo "   Please install Python 3.8 or later from python.org"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    
    # Install the package
    echo "📥 Installing WhisperLive Transcription..."
    pip install --upgrade pip
    pip install .
    
    # Install local WhisperLiveKit with our fixes
    if [ -d "whisperlivekit" ]; then
        echo "🔧 Installing local WhisperLiveKit with fixes..."
        cd whisperlivekit
        pip install -e .
        cd ..
    fi
    
    echo ""
    echo "✅ Setup complete!"
    
    echo ""
    echo "To start WhisperLive Transcription:"
    echo "  ./run.sh"
    exit 0
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "It looks like this is your first time running WhisperLive."
    echo "Please run setup first:"
    echo ""
    echo "  ./run.sh --setup"
    echo ""
    exit 1
fi

echo "🎙️  Starting WhisperLive Transcription..."

# Kill any existing WhisperLiveKit servers
echo "Checking for existing WhisperLiveKit servers..."
EXISTING_PIDS=$(lsof -ti:9090 2>/dev/null || true)
if [ ! -z "$EXISTING_PIDS" ]; then
    echo "Found existing process on port 9090, terminating..."
    kill -9 $EXISTING_PIDS 2>/dev/null || true
    sleep 1
fi

# Also kill any whisperlivekit-server processes
pkill -f "whisperlivekit-server" 2>/dev/null || true

# Run in background, detached from terminal
nohup ./venv/bin/python -m whisper_transcriber.main > /dev/null 2>&1 &

# Get the process ID
PID=$!

# Give it a moment to start
sleep 2

# Check if it's running
if ps -p $PID > /dev/null; then
    echo "✅ WhisperLive is running in the background (PID: $PID)"
    echo "   Look for the microphone icon in your menu bar"
    
    echo ""
    echo "To stop WhisperLive:"
    echo "   - Click the menu bar icon → Quit"
    echo "   - Or run: kill $PID"
else
    echo "❌ Failed to start WhisperLive"
    exit 1
fi