#!/usr/bin/env bash
set -e

echo "==================================================="
echo "Setting up Lab 1 Virtual Environment (Python 3.11+)"
echo "==================================================="

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
pip install -r requirements.txt

python3 -m spacy download en_core_web_sm
python3 -m ipykernel install --user --name nlp-llm-lab1 --display-name "Python (NLP-LLM Lab1)"

echo "==================================================="
echo "Lab 1 Environment Setup Complete!"
echo "Launch with: jupyter notebook lab1.ipynb"
echo "==================================================="
