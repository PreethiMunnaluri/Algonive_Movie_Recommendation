from flask import Flask, render_template, request, jsonify
from model import recommender

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', movies=recommender.get_movie_titles())

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie']
    recommendations = recommender.get_recommendations(movie_title)
    
    if recommendations:
        return jsonify({
            "status": "found",
            "recommendations": recommendations
        })
    else:
        similar = recommender.suggest_similar_titles(movie_title)
        fallback = recommender.fallback_recommendations()
        return jsonify({
            "status": "not_found",
            "similar_titles": similar,
            "recommendations": fallback
        })

if __name__ == '__main__':
    app.run(debug=True)