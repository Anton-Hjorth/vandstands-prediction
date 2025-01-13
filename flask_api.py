from flask import Flask, jsonify, request # type: ignore

app = Flask(__name__)

# Sample data to simulate a database
items = [
    {"id": 1, "name": "Item 1", "description": "This is item 1"},
    {"id": 2, "name": "Item 2", "description": "This is item 2"},
]

# Home route
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

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

# Add a new item
@app.route('/api/items', methods=['POST'])
def add_item():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Invalid input"}), 400

    new_item = {
        "id": items[-1]["id"] + 1 if items else 1,
        "name": data["name"],
        "description": data.get("description", "")
    }
    items.append(new_item)
    return jsonify(new_item), 201

# Update an existing item
@app.route('/api/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    data = request.get_json()
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"error": "Item not found"}), 404

    item["name"] = data.get("name", item["name"])
    item["description"] = data.get("description", item["description"])
    return jsonify(item)

# Delete an item
@app.route('/api/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global items
    items = [item for item in items if item["id"] != item_id]
    return jsonify({"message": "Item deleted"})

if __name__ == '__main__':
    app.run(debug=True)
