import argparse,json
from pathlib import Path
import joblib
from .model import train,predict
from .synthetic import FEATURES
def main():
    p=argparse.ArgumentParser();p.add_argument("--samples",type=int,default=1200);p.add_argument("--alpha",type=float,default=.1);p.add_argument("--out",default="artifacts")
    a=p.parse_args();model=train(n=a.samples,alpha=a.alpha)
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,folder/"model.joblib")
    (folder/"metrics.json").write_text(json.dumps(model.metrics,indent=2))
    print(json.dumps(model.metrics,indent=2))
    print("Example:",predict(model,{f:2 for f in FEATURES}))
if __name__=="__main__":main()
