from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@cross_origin()
@app.route('/', methods=['POST'])
def analyze():
    model = request.json.get("model", None)
    text = request.json.get("text", None)

    if(model == "Multinomial Naive Bayes"):
        return predictWithModelAndPickle(text,
                                         "models/mnb_trained_model.pkl", 'models/vectorizers/mnb_vectorizer.pickle')

    if(model == "K-Nearest Neighbor"):
        return predictWithModelAndPickle(text,
                                         'models/knn_trained_model.pkl', 'models/vectorizers/knn_vectorizer.pickle')
    if(model == "Support Vector Machine"):
        return predictWithModelAndPickle(text,
                                         'models/svm_trained_model.pkl', 'models/vectorizers/svm_vectorizer.pickle')


def predictWithModelAndPickle(text, pickleFile, modelFile):
    analysis_result = {}
    loaded_model = pickle.load(open(pickleFile, 'rb'))
    tfidf = pickle.load(
        open(modelFile, 'rb'))
    sample_text = []
    sample_text.append(text)
    transformed_text = tfidf.transform(sample_text)
    print(transformed_text)
    print(transformed_text.shape)

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
