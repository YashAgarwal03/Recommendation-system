from flask import Flask, render_template, request
from recommendation_funcation import content_based,collaborative_filtering, hybrid_recommendation,dataset

app = Flask(__name__)

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/content', methods=['GET', 'POST'])
def content_page():
    # Retrieve unique product names
    product_names = dataset['product_name'].unique()
    
    if request.method == 'POST':
        product_name = request.form.get('product_name')
        results = content_based(product_name)
        return render_template(
            'results.html', 
            tables=[results.to_html(classes='table')], 
            title="Content-Based Recommendation Results"
        )
    
    return render_template('content.html', product_names=product_names)


@app.route('/collaborative', methods=['GET', 'POST'])
def collaborative_page():
    # Retrieve unique user IDs
    user_ids = dataset['user_id'].unique()
    
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        results = collaborative_filtering(user_id)
        return render_template(
            'results.html', 
            tables=[results.to_html(classes='table')], 
            title="Collaborative Filtering Recommendation Results"
        )
    
    return render_template('collaborative.html', user_ids=user_ids)


@app.route('/hybrid', methods=['GET', 'POST'])
def hybrid_page():
    # Retrieve unique user IDs and product names
    user_ids = dataset['user_id'].unique()
    product_names = dataset['product_name'].unique()

    if request.method == 'POST':
        user_id = request.form.get('user_id')
        product_name = request.form.get('product_name')
        results = hybrid_recommendation(user_id, product_name)
        return render_template(
            'results.html', 
            tables=[results.to_html(classes='table')], 
            title="Hybrid Recommendation Results"
        )
    
    return render_template('hybrid.html', user_ids=user_ids, product_names=product_names)


if __name__ == '__main__':
    app.run(debug=True)
