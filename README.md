# StoryPoint Uncertainty Lab — Machine Learning for Agile Estimation

Train an educational **Random Forest with split-conformal prediction intervals** on synthetic Agile story complexity features. Evaluate calibration/test coverage, MAE, and interval widths. This is related to software-estimation research, not an implementation of private university research datasets.

## Run
```bash
pip install -e '.[test]'
python -m storypoint.cli --samples 1000
pytest -q
```
Writes `artifacts/model.joblib` and `artifacts/metrics.json`. Entirely synthetic data; real team performance and calibration would require prospective validation.