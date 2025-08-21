[README_cleaning_section.md](https://github.com/user-attachments/files/21908198/README_cleaning_section.md)
## Data Cleaning & Preprocessing (Stage 6)

This stage turns the in-class patterns for filling, dropping, and scaling into **reusable, documented utilities** and applies them to the raw dataset.

### What’s Included
- `src/cleaning.py`: modular functions
  - `fill_missing_median(df, columns=None)`
  - `drop_missing(df, how="any", thresh=None, subset=None)`
  - `normalize_data(df, columns=None, method="standard"|"minmax")`
- `notebooks/stage06_preprocessing.ipynb`: loads a CSV from `data/raw/`, applies cleaning, compares original vs cleaned, and saves to `data/processed/`.

### How to Run
1. Place raw CSV(s) in `data/raw/`.
2. Open and run `notebooks/stage06_preprocessing.ipynb` top-to-bottom.
3. The cleaned dataset is saved as `<rawname>_cleaned.csv` in `data/processed/`.

### Assumptions & Rationale
- **Filling**: Numeric NaNs are filled with column medians to minimize information loss.
- **Dropping**: After filling, remaining missing rows are dropped to simplify downstream modeling.
- **Scaling**: Default is **standardization**; switch to **min-max** if your model or domain requires a bounded range.
- **Reproducibility**: All functions are pure (return new dataframes). `normalize_data` also returns fitted parameters per column; persist these if you need to transform future/holdout data exactly the same way.

### Reproducibility Checklist
-  Notebook runs end-to-end without manual edits
-  Cleaned file saved to `data/processed/`
-  Functions have docstrings & clear parameters
-  README explains design choices

### Submission Artifacts
- `src/cleaning.py`
- `notebooks/stage06_preprocessing.ipynb`
- Updated `README.md` with this section
- Cleaned dataset in `data/processed/`
