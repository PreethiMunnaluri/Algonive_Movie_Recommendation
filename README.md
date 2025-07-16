# CineMatch 🎥  
An AI-powered movie recommendation web app where users can search by title, actor, theme, or vibe—and get instant suggestions based on movie similarity.

## 🚀 Features
- 🎯 Intelligent recommendations powered by TF-IDF and cosine similarity
- 🔍 Search by mood, genre, actor, or title
- 🧠 Clickable suggestion buttons for quick queries
- 💖 Responsive UI with dark mode styling
- 🛠️ Fallback logic for unmatched searches with fuzzy title suggestions
- ⚡ Real-time recommendations via AJAX

## 🖼️ Technologies Used
- Python & Flask
- Pandas & Scikit-learn
- HTML, CSS, JavaScript, jQuery
- TMDB 5000 Movies + Credits dataset


## 🧪 How to Run Locally

### Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Run the app
python app.py
Then open http://127.0.0.1:5000 in your browser
