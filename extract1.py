from typing import Optional
import pydantic


class Experience(pydantic.BaseModel):
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]


class Study(Experience):
    degree: Optional[str]
    university: Optional[str]
    country: Optional[str]
    grade: Optional[str]


class WorkExperience(Experience):
    company: str
    job_title: str


class Resume(pydantic.BaseModel):
    first_name: str
    last_name: str
    linkedin_url: Optional[str]
    email_address: Optional[str]
    nationality: Optional[str]
    skill: Optional[str]
    study: Optional[Study]
    work_experience: Optional[WorkExperience]
    hobby: Optional[str]
