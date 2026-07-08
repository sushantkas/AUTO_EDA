# AUTO_EDA

Auto EDA Studio is a Streamlit application for automated exploratory data analysis, feature engineering, visualization, and hypothesis testing.

## Requirements

This project supports Python 3.11 and depends on the packages listed in `environment.yml` and `requirements.txt`.

## Setup

### Option 1: Create a Conda environment

```bash
conda env create -f environment.yml
conda activate eda_studio
```

### Option 2: Use pip in a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

From the project root (`E:\Auto_EDA`):

```bash
streamlit run auto_eda_app.py
```

Then open the URL shown in the terminal, usually `http://localhost:8501`.

## Notes

- If you want to analyze your own dataset, upload it through the Streamlit app interface.
- The app uses optional profiling libraries (`sweetviz`, `ydata-profiling`, `streamlit-pandas-profiling`) and degrades gracefully if any are unavailable.
- The project includes a `.gitignore` file to exclude common temporary files and virtual environments.
