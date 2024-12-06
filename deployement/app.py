from flask import Flask, render_template, request
from recommendation_funcation import content_based,collaborative_filtering,hybird_recomend

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/function_one', methods=['GET', 'POST'])
def run_function_one():
    if request.method == 'POST':
        product_name = request.form['product_name']
        result = content_based(product_name)
        return render_template('function_one.html', result=result.to_html())
    return render_template('function_one.html')

@app.route('/function_two', methods=['GET', 'POST'])
def run_function_two():
    if request.method == 'POST':
        user_id = request.form['user_id']
        result = collaborative_filtering(user_id)
        return render_template('function_two.html', result=result.to_html())
    return render_template('function_two.html')

@app.route('/function_three', methods=['GET', 'POST'])
def run_function_three():
    if request.method == 'POST':
        product_name = request.form['product_name']
        user_id = request.form['user_id']
        result = hybird_recomend(user_id,product_name)
        return render_template('function_three.html', result=result.to_html())
    return render_template('function_three.html')

if __name__ == '__main__':
    app.run(debug=True)
