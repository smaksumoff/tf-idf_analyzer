# TF-IDF Analyzer

This is a simple Flask-based web application that allows users to upload a text file and compute TF-IDF values for the 
words in the text. The app supports paginated results and can be reset to handle new files.

## Screenshot
Here is how the web application looks in the browser:

![TF-IDF Web Page Screenshot](screenshots/webpage_screenshot.png)

## Features
- Upload `.txt` files to analyze
- Tokenizes text into words and sentences
- Computes Term Frequency (TF) and Inverse Document Frequency (IDF)
- Displays top 50 words with their TF and IDF values
- Pagination for the results
- Reset functionality to clear uploaded files/session data

## Requirements
- Python 3.8 or higher
- Flask
- Any modern browser for accessing the web app

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/smaksumoff/tf-idf_analyzer.git
   cd tfidf-analyzer
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Start the Flask server:
   ```bash
   python app.py
4. Open your browser and navigate to:
   ```bash
   http://127.0.0.1:5000

## Usage

1. Click the "**Upload**" button to upload a `.txt` file.
2. View the TF-IDF analysis and navigate pages using "**Previous**" and "**Next**".
3. Click "**Reset**" to clear results and start over.

## 