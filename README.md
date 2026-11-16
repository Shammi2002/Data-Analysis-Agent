# Telecom Data Analysis Agent


https://github.com/user-attachments/assets/9cec5fa9-33e6-4d7a-b889-5385d40ae3a6


A Streamlit app where you upload CSV files and ask questions about them in plain language.
A LangGraph loop sends your question to a Groq-hosted model, which writes Python (Polars + Plotly),
runs it on your data and answers with numbers, charts or a PDF report.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY="your-key"   # Windows PowerShell: $env:GROQ_API_KEY="your-key"
streamlit run app.py
```

Then upload a CSV, select it, press **Go** and start asking questions.
Charts inside PDF reports need `kaleido` (included in requirements.txt).

## Project layout

| File | What it does |
| --- | --- |
| `app.py` | Streamlit UI and entry point: upload, dataset selection, chat window |
| `chatbot.py` | `PythonChatbot`: keeps the conversation and links charts/PDFs to answers |
| `agent/graph.py` | LangGraph nodes (`call_model`, `call_tools`), routing and `build_graph()` |
| `agent/tools.py` | `complete_python_task` tool: runs the model's code, saves charts, checks PDFs |
| `reports/pdf.py` | `make_pdf_report`, `numeric_summary`, `category_summary` |
| `llm.py` | Creates the Groq chat model (reads `GROQ_API_KEY`) |
| `prompts.py` | The system prompt |
| `data_utils.py` | State types, `describe_df` and the per-turn dataset summary |
| `config.py` | Folders, model settings and limits |

Uploaded files, charts and PDFs are stored in `uploaded_datasets/`, `images/` and `generated_pdfs/`,
which are created next to the code on first run.
