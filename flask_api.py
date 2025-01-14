from flask import Flask, jsonify, request # type: ignore
import requests


app = Flask(__name__)

# Sample data to simulate a database
items = [
    {"id": 1, "name": "Item 1", "description": "This is item 1"},
    {"id": 2, "name": "Item 2", "description": "This is item 2"},
]

# Home route
@app.route('/')
def home():
    return jsonify([{"Velkommen": "Kammersluse vandstands-API"},
                    {"GET links": "/api/items"}])

# Get all items
@app.route('/api/items', methods=['GET'])
def get_items():
    return jsonify(items)

# Get a single item by ID
@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(item)


if __name__ == '__main__':
    app.run(debug=True)
