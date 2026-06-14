# hse_python_project
a repo for final python project in data analysis done during first-year studies at hse


# Installment

## 1 - clone repository
```bash
   git clone https://github.com/HerrHeiler/hse_python_project.git
   cd hse_python_project
```

## 2 - create and activate virtual environment
- MacOs/Linux
```bash
   python3 -m venv .venv
   source .venv/bin/activate
```
- Windows
```bash
   python -m venv .venv
   .venv\Scripts\activate
```
## 3 - install requirements
- MacOs/Linux
```bash
   pip3 install -r requirements.txt
```
- Windows
```bash
   pip install -r requirements.txt
```

# Launch - from root

## - Streamlit
```bash
  streamlit run apps/streamlit_app.py
```
go to http://localhost:8501/
## - FastAPI
```bash
  uvicorn apps.api:app --reload --port 8000
```
go to http://127.0.0.1:8000/docs#/ for swagger

## Note

### API Sleep Mode (Free Tier)

The API is hosted on Render's free tier which has some limitations where API falls asleep after 15 mins of inactivity

What to do:
If you see "API is not running" - wait 30-60 seconds and refresh
(or open `/docs` endpoint to wake up the API manually)

Endpoints:
- API: https://student-api-hse.onrender.com
- Swagger Docs: https://student-api-hse.onrender.com/docs
- Dashboard: https://student-dashboard-xyz.onrender.com