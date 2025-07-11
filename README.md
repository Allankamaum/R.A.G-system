# RAG-Pull

This is a small app that lets you ask questions about a `.txt` file using AI.

## What It Does

- You upload a `.txt` file.
- It breaks the file into smaller pieces (chunks).
- It turns those chunks into something the AI can understand (embeddings).
- Then you ask a question, and the AI tries to find the answer based on your file.

## How It Works

- Uses Langchain to split the file.
- Uses Chroma to store and search the text.
- Uses a local embedding model (no `## OpenAI` for current app version) to keep it free.
- Answers are generated using a small QA chain.

## File Flow

- Uploaded files go to `app/uploaded-files/`
- Vector database goes to `app/chroma_db/`

## How to Run

Make sure you have Python set up and install the `requirements`.Then:

```bash
streamlit run app/main.py
```

That opens the app in your browser.

## Notes

- Still being worked on, so expect bugs.
- More features coming later maybe.
