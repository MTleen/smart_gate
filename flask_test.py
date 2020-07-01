from flask import Flask
import requests
import json

app = Flask(__name__)


@app.route('/states')
def detect():
    return json.dumps({'is_face': False})


if __name__ == '__main__':
    app.run()
