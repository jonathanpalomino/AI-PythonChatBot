#!/bin/bash
# =============================================================================
# scripts/setup.sh
# Complete setup script for RAG Chatbot
# =============================================================================

set -e  # Exit on error

# Change to project root directory
cd "$(dirname "$0")/.."

echo "================================================"
echo "RAG Chatbot - Complete Setup"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo -e "${RED}❌ Python 3.11 or higher is required${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"
echo ""

# Check Docker
echo "📋 Checking Docker..."
if ! docker --version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"
echo ""

# Create virtual environment
echo "🐍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
. venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Setup .env file
echo "⚙️  Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✅ .env file created from template${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env with your configuration${NC}"
else
    echo -e "${YELLOW}⚠️  .env already exists, skipping${NC}"
fi
echo ""

# Start Docker services
echo "🐳 Starting Docker services..."
docker compose up -d
echo "⏳ Waiting for services to be ready..."
sleep 10
echo -e "${GREEN}✅ Docker services started${NC}"
echo ""

# Check Ollama
echo "🤖 Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama is installed${NC}"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Ollama is not running. Start it with: ollama serve${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Ollama not found. Install from: https://ollama.com${NC}"
fi
echo ""

# Initialize database
#echo "🗄️  Initializing database..."
#python scripts/init_db.py
#echo -e "${GREEN}✅ Database initialized${NC}"
#echo ""

# Summary
echo "================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Start the API: make run or python main.py"
echo "2. Test the API: make test-api or python scripts/test_api.py"
echo "3. Access docs: http://localhost:8000/api/v1/docs"
echo ""
echo "Optional:"
echo "- Install Ollama models: ollama pull mistral && ollama pull mxbai-embed-large"
echo "- Configure API keys in .env for remote providers"
echo ""
