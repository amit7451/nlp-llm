@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Setting up Global NLP & LLM Virtual Environment
echo ===================================================

REM Check Python availability
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.11+ and ensure it is added to your PATH.
    pause
    exit /b 1
)

REM Create virtual environment if it does not exist
if not exist ".venv" (
    echo Creating virtual environment in .venv...
    python -m venv .venv
) else (
    echo Existing .venv found.
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Downloading NLTK standard data assets...
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words', 'stopwords', 'wordnet', 'omw-1.4']]"

echo Downloading spaCy english model...
python -m spacy download en_core_web_sm

echo Registering Jupyter kernel 'nlp-llm'...
python -m ipykernel install --user --name nlp-llm --display-name "Python (NLP-LLM)"

echo ===================================================
echo Global NLP & LLM Environment Setup Complete!
echo Kernel Registered: 'Python (NLP-LLM)'
echo Activate manually via:
echo   Windows: .\.venv\Scripts\activate
echo   Linux/macOS: source .venv/bin/activate
echo ===================================================
pause
