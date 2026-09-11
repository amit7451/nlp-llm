#!/usr/bin/env bash
set -e

echo "==================================================="
echo "Setting up Global NLP & LLM Virtual Environment"
echo "==================================================="

# Detect python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Neither python3 nor python was found in PATH."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    $PYTHON_CMD -m venv .venv
else
    echo "Existing .venv found."
fi

# Activate environment
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Downloading NLTK standard data assets..."
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words', 'stopwords', 'wordnet', 'omw-1.4']]"

echo "Downloading spaCy english model..."
python -m spacy download en_core_web_sm

echo "Registering Jupyter kernel 'nlp-llm'..."
python -m ipykernel install --user --name nlp-llm --display-name "Python (NLP-LLM)"

echo "==================================================="
echo "Global NLP & LLM Environment Setup Complete!"
echo "Kernel Registered: 'Python (NLP-LLM)'"
echo "Activate manually via: source .venv/bin/activate"
echo "==================================================="
