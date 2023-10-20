import pandas as pd
import pickle
from flask import Flask, jsonify, request

with open('Best_model.pkl', 'rb') as file:
    load_model = pickle.load(file)

with open('OHE.pkl', 'rb') as file:
    load_ohe = pickle.load(file)

with open('SVD.pkl', 'rb') as file:
    load_svd = pickle.load(file)

app = Flask('default')


@app.route('/predict', methods=['POST'])
def predict():
    X_dict = request.get_json()
    X = pd.DataFrame(X_dict, index=[0])
    one_hot_encoded = load_ohe.transform(X['Destination Port'].values.reshape(1, -1))
    embeddings = load_svd.transform(one_hot_encoded)
    embedding_cols = [f'embedding_{i + 1}' for i in range(embeddings.shape[1])]

    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    data_with_embeddings = pd.concat([embedding_df, X.drop('Destination Port', axis=1)], axis=1)
    predictions = load_model.predict(data_with_embeddings)[0]
    result = {'Prediction': predictions}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
