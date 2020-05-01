from flask import Flask
from flask import render_template
from flask import request

from deployment import get_flair_from_urlids

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    urlid = request.form['urlid']
    prediction = get_flair_from_urlids([urlid])
    return render_template(
        'result.html',
        flair=prediction[0]
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0')
