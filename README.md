# hse_python_project
a repo for final python project in data analysis done during first-year studies at hse


## Installment

# 1 - clone repository
```bash
   git clone https://github.com/HerrHeiler/hse_python_project.git
   cd hse_python_projet
```

# 2 - create and activate virtual environment
```bash
   python3 -m venv .venv
   source .venv/bin/activate
```

# 3 - install requirements
```bash
   pip3 install -r requirements.txt
```

## Launch

# - Streamlit
```bash
  streamlit run apps/streamlit_app.py
```

# - FastAPI
```bash
  uvicorn apps.api:app --reload
```