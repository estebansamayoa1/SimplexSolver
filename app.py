from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def solver():
    numeros=request.args
    if(numeros):
        print(numeros)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)