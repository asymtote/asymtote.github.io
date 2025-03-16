from flask import Flask , render_template , url_for, request, redirect
from factcheck import manual_testing 
app=Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        article = request.form['content']
        model=request.form['model']
        res = manual_testing(article,model)  
    else:
       return render_template('index.html', title='Misinformation Detection')

@app.route('/performance')
def performance():
    return render_template('performance.html',title='Performance Analysis')
if __name__ == "__main__":
    app.run(debug=True)