# dashboard.py
import json
from pathlib import Path

import pandas as pd
import streamlit as st

EXPERIMENTS_DIR = Path(__file__).parent.parent / 'experiments'

st.set_page_config(page_title='Experiment Progress', layout='wide')
st.title('Hyperparameter Search Dashboard')

experiment_names = sorted(
    p.name for p in EXPERIMENTS_DIR.iterdir()
    if p.is_dir() and (p / 'results.csv').exists()
)

if not experiment_names:
    st.warning(f'No experiments found under {EXPERIMENTS_DIR}. '
              f'Run `python src/experiment.py init ...` first.')
    st.stop()

selected = st.sidebar.selectbox('Experiment', experiment_names)
exp_dir = EXPERIMENTS_DIR / selected

df = pd.read_csv(exp_dir / 'results.csv')
df.insert(0, 'generation', range(len(df)))

st.subheader(f'{selected} — current best')
best_rows = df[df['is_new_best']]
if best_rows.empty:
    st.info('No generation has been recorded as best yet.')
else:
    best_run = best_rows.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric('Overall AUC', f"{best_run['overall_auc']:.4f}")
    col2.metric('Windowed AUC avg', f"{best_run['windowed_auc_avg']:.4f}")
    col3.metric('Log loss', f"{best_run['overall_log_loss']:.4f}")

st.subheader('AUC over generations')
chart_df = df[['generation', 'overall_auc', 'windowed_auc_avg']].set_index('generation')
st.line_chart(chart_df)

st.subheader('Experiment log')
st.dataframe(
    df[['generation', 'generation_file', 'hypothesis', 'is_new_best',
       'overall_auc', 'windowed_auc_avg', 'overall_log_loss']],
    width='stretch',
)

st.subheader('Best config')
best_json_path = exp_dir / 'best.json'
if best_json_path.exists():
    st.json(json.loads(best_json_path.read_text()))
else:
    st.info('best.json not found.')
