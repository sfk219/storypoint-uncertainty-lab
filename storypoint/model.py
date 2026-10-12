"""Split-conformal random-forest uncertainty interval for synthetic Agile story points."""
from dataclasses import dataclass
from math import ceil
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from .synthetic import generate,FEATURES
@dataclass
class Result:
    model:RandomForestRegressor
    cutoff:float
    alpha:float
    metrics:dict

def train(n=1200,seed=42,alpha=.1):
    if not 0<alpha<1:raise ValueError("alpha must lie between 0 and 1")
    data=generate(n,seed)
    x,y=data[FEATURES],data["story_points"]
    xd,xt,yd,yt=train_test_split(x,y,test_size=.2,random_state=seed)
    xtr,xc,ytr,yc=train_test_split(xd,yd,test_size=.25,random_state=seed+1)
    model=RandomForestRegressor(n_estimators=100,min_samples_leaf=4,random_state=seed,n_jobs=1)
    model.fit(xtr,ytr)
    residuals=np.abs(yc.to_numpy()-model.predict(xc))
    rank=min(len(residuals),ceil((len(residuals)+1)*(1-alpha)))
    cutoff=float(np.sort(residuals)[rank-1])
    prediction=model.predict(xt)
    lo=np.maximum(1,prediction-cutoff);hi=np.minimum(13,prediction+cutoff)
    metrics={"synthetic":True,"train_rows":len(ytr),"calibration_rows":len(yc),"test_rows":len(yt),
             "target_coverage":1-alpha,
             "empirical_coverage":round(float(np.mean((yt.to_numpy()>=lo)&(yt.to_numpy()<=hi))),3),
             "test_mae":round(float(mean_absolute_error(yt,prediction)),3),
             "mean_interval_width":round(float(np.mean(hi-lo)),3)}
    return Result(model,cutoff,alpha,metrics)

def predict(result,features):
    if set(features)!=set(FEATURES):raise ValueError("Input must match all six features")
    import pandas as pd
    x=pd.DataFrame([{k:features[k] for k in FEATURES}])
    value=float(result.model.predict(x)[0])
    return {"estimate":round(value,2),"lower":round(max(1,value-result.cutoff),2),
            "upper":round(min(13,value+result.cutoff),2),"target_coverage":round(1-result.alpha,2)}
