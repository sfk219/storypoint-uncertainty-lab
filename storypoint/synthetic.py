"""Entirely synthetic Agile story feature examples."""
import numpy as np
import pandas as pd
FEATURES=["acceptance_criteria","dependencies","technical_complexity","requirements_ambiguity","integration_points","team_familiarity"]
POINTS=np.array([1,2,3,5,8,13],dtype=float)
def generate(n=1200,seed=42):
    if n<120:raise ValueError("At least 120 samples required")
    rng=np.random.default_rng(seed)
    frame=pd.DataFrame({
        "acceptance_criteria":rng.integers(1,12,n),
        "dependencies":rng.integers(0,6,n),
        "technical_complexity":rng.integers(1,6,n),
        "requirements_ambiguity":rng.integers(0,6,n),
        "integration_points":rng.integers(0,5,n),
        "team_familiarity":rng.integers(1,6,n)})
    s=(.3*frame.acceptance_criteria+.8*frame.dependencies+1.4*frame.technical_complexity+
       .6*frame.requirements_ambiguity+.7*frame.integration_points-.8*frame.team_familiarity)
    noise=s.to_numpy()+rng.normal(0,2,n)
    frame["story_points"]=POINTS[np.abs(noise[:,None]-POINTS[None,:]).argmin(axis=1)]
    return frame
