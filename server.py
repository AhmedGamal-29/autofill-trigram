from trigram import *
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('search_view.html')


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    corpus = collect_corpus()
    preprepared_text = prepare_text(corpus)
    # Create trigram model and n-1 gram model
    n = 3
    ngram_model = create_ngram_model(preprepared_text, n)
    n_minus1_gram = create_ngram_model(preprepared_text, n - 1)
    partial = request.args.get('partial')
    context = tuple(partial.split()[-(n - 1):])  # Get last (n-1) words from partial input
    suggestions = predict_word(ngram_model, n_minus1_gram, context)
    return jsonify(suggestions)