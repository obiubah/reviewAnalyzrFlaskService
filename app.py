from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
app = Flask(__name__)


@app.route('/', methods=['POST'])
def analyze():
    model = request.json.get("model", None)
    text = request.json.get("text", None)

    if(model == "Multinomial Naive Bayes"):
        analysis_result = {}
        loaded_model = pickle.load(open('models/mnb_trained_model.pkl', 'rb'))
        tfidf = pickle.load(
            open("models/vectorizers/mnb_vectorizer.pickle", 'rb'))
        sample_text = []
        sample_text.append(text)
        transformed_text = tfidf.transform(sample_text)
        print(transformed_text)

        prediction = loaded_model.predict(transformed_text)

        if(prediction == 0):
            analysis_result["genuine"] = "FALSE"
        else:
            analysis_result["genuine"] = "TRUE"

        analysis_result["accuracy"] = np.max(
            loaded_model.predict_proba(transformed_text))*100

        return jsonify(analysis_result)


if __name__ == '__main__':
    app.run()
