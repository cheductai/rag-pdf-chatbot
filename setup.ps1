# RAG PDF Chatbot - Quick Start Script for Windows
# This script helps you set up and run the RAG PDF Chatbot

Write-Host "🚀 RAG PDF Chatbot - Quick Start" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ from python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Write-Host "💡 Try running: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✅ Created .env file from example" -ForegroundColor Green
        Write-Host "⚠️  Please edit .env file and add your OpenAI API key!" -ForegroundColor Yellow
    } else {
        Write-Host "❌ .env.example file not found" -ForegroundColor Red
    }
} else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}

# Create necessary directories
$directories = @("data\uploads", "data\processed", "data\faiss_index")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "✅ Created necessary directories" -ForegroundColor Green

Write-Host "`n🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file and add your OpenAI API key" -ForegroundColor White
Write-Host "2. Run the application: python main.py" -ForegroundColor White
Write-Host "`n💡 Make sure you have an OpenAI API key!" -ForegroundColor Yellow

# Ask if user wants to run the application
$runApp = Read-Host "`nWould you like to run the application now? (y/N)"
if ($runApp -eq "y" -or $runApp -eq "Y") {
    Write-Host "`n🚀 Starting RAG PDF Chatbot..." -ForegroundColor Green
    python main.py
}
