@echo off
echo ===================================================
echo Setting up Lab 1 Virtual Environment (Python 3.11+)
echo ===================================================

python -m venv .venv
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -m ipykernel install --user --name nlp-llm-lab1 --display-name "Python (NLP-LLM Lab1)"

echo ===================================================
echo Lab 1 Environment Setup Complete!
echo Launch with: jupyter notebook lab1.ipynb
echo ===================================================
pause
