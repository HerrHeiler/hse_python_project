from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Student Lifestyle API")

BASE_DIR = Path.cwd()
RAW_PATH = BASE_DIR / "data" / "raw" / "student_lifestyle_dataset.csv"
NEW_ENTRIES_PATH = BASE_DIR / "data" / "new_entries" / "new_students.csv"

df = pd.read_csv(RAW_PATH)


class StudentRecord(BaseModel):
    Study_Hours_Per_Day: float
    Extracurricular_Hours_Per_Day: float
    Sleep_Hours_Per_Day: float
    Social_Hours_Per_Day: float
    Physical_Activity_Hours_Per_Day: float
    Stress_Level: str
    Gender: str
    Grades: float


def get_next_id():
    max_id = int(df["Student_ID"].max())

    if NEW_ENTRIES_PATH.exists():
        df_new = pd.read_csv(NEW_ENTRIES_PATH)
        max_id = max(max_id, int(df_new["Student_ID"].max()))

    return max_id + 1


@app.get("/students")
def get_students(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Records per page"),
    gender: str = Query(None, description="Filter by gender"),
    stress_level: str = Query(None, description="Filter by stress level"),
    min_grades: float = Query(None, ge=0, le=10, description="Minimum grades"),
    max_grades: float = Query(None, ge=0, le=10, description="Maximum grades"),
    include_new: bool = Query(True, description="Include new entries?"),
):
    filtered_df = df.copy()

    if include_new and NEW_ENTRIES_PATH.exists():
        df_new = pd.read_csv(NEW_ENTRIES_PATH)
        filtered_df = pd.concat([filtered_df, df_new], ignore_index=True)

    if gender:
        filtered_df = filtered_df[filtered_df["Gender"] == gender]
    if stress_level:
        filtered_df = filtered_df[filtered_df["Stress_Level"] == stress_level]
    if min_grades is not None:
        filtered_df = filtered_df[filtered_df["Grades"] >= min_grades]
    if max_grades is not None:
        filtered_df = filtered_df[filtered_df["Grades"] <= max_grades]

    total_records = len(filtered_df)
    total_pages = (total_records + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    paginated_df = filtered_df.iloc[start_idx:end_idx]

    return {
        "total_records": total_records,
        "raw_records": len(df),
        "new_records": len(pd.read_csv(NEW_ENTRIES_PATH))
        if NEW_ENTRIES_PATH.exists()
        else 0,
        "total_pages": total_pages,
        "current_page": page,
        "page_size": page_size,
        "data": paginated_df.to_dict(orient="records"),
    }


@app.post("/students")
def create_student(student: StudentRecord):

    if student.Grades < 0 or student.Grades > 10:
        raise HTTPException(status_code=400, detail="Grades must be between 0 and 10")

    if student.Stress_Level not in ["Low", "Moderate", "High"]:
        raise HTTPException(
            status_code=400, detail="Stress_Level must be Low, Moderate, or High"
        )

    new_id = get_next_id()

    new_record = pd.DataFrame([{"Student_ID": new_id, **student.dict()}])

    if NEW_ENTRIES_PATH.exists():
        df_new = pd.read_csv(NEW_ENTRIES_PATH)
        df_new = pd.concat([df_new, new_record], ignore_index=True)
        df_new.to_csv(NEW_ENTRIES_PATH, index=False)
    else:
        new_record.to_csv(NEW_ENTRIES_PATH, index=False)

    return {
        "message": "Student record created successfully",
        "student_id": new_id,
        "location": "data/new_entries/new_students.csv",
        "data": student.dict(),
    }


@app.get("/statistics")
def get_statistics():
    combined_df = df.copy()

    if NEW_ENTRIES_PATH.exists():
        df_new = pd.read_csv(NEW_ENTRIES_PATH)
        combined_df = pd.concat([combined_df, df_new], ignore_index=True)

    return {
        "total_students": len(combined_df),
        "raw_students": len(df),
        "new_students": len(pd.read_csv(NEW_ENTRIES_PATH))
        if NEW_ENTRIES_PATH.exists()
        else 0,
        "average_grades": combined_df["Grades"].mean(),
        "stress_distribution": combined_df["Stress_Level"].value_counts().to_dict(),
        "gender_distribution": combined_df["Gender"].value_counts().to_dict(),
    }


@app.delete("/students/clear-new")
def clear_new_entries():
    if NEW_ENTRIES_PATH.exists():
        NEW_ENTRIES_PATH.unlink()
        return {"message": "All new entries cleared"}
    else:
        return {"message": "No new entries to clear"}
