from flask import Flask, jsonify, request # type: ignore
import requests
import json


app = Flask(__name__)




# Home route
@app.route('/')
def home():
    return jsonify([{"Velkommen": "Kammersluse vandstands-API"},
                    {"GET links": "/api/data"}])


@app.route('/api/data', methods=['GET'])
def get_items():
    with open('data_for_api.json', 'r') as openfile:
        loaded_data = json.load(openfile)

        return jsonify(loaded_data)




if __name__ == '__main__':
    app.run(debug=True)
