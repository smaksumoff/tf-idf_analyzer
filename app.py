from flask import Flask, request, render_template, session, redirect, url_for
from typing import List, Dict, Any
from config import SECRET_KEY
from utils.tf_idf_processor import TfIdfProcessor

# Flask app initialization
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Pagination configuration
ROWS_PER_PAGE: int = 10


@app.route('/', methods=["GET", "POST"])
def index() -> Any:
    """
    Handle file upload and page rendering

    :return: Rendered HTML template for the index page.
    """
    result_table: List[Dict[str, Any]] = []
    total_pages: int = 0
    current_page: int = int(request.args.get('page', 1))

    if request.method == "POST":
        # Handle file upload
        file = request.files['file']
        document: str = file.read().decode('utf-8')

        # Process the TF and IDF results
        processor: TfIdfProcessor = TfIdfProcessor()
        result_table = processor.compute_tfidf(document)
        session['result_table'] = result_table

        total_pages = (len(result_table) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
        session['total_pages'] = total_pages

    elif request.method == "GET":
        # Retrieve table and pagination info from session
        result_table = session.get('result_table', [])
        total_pages = session.get('total_pages', 0)

    # Pagination logic
    start_row: int = (current_page - 1) * ROWS_PER_PAGE
    end_row: int = start_row + ROWS_PER_PAGE
    paginated_table: List[Dict[str, Any]] = result_table[start_row:end_row]

    return render_template(
        "index.html",
        result_table=paginated_table,
        total_pages=total_pages,
        current_page=current_page
    )


@app.route('/reset')
def reset() -> Any:
    """
    Clear session data and redirect to the index route.

    :return: Redirect to the index route.
    """
    session.clear()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
