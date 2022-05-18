from flask import Flask, render_template, request
import numpy as np 
from fractions import Fraction
import sympy as sp


app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def solver():
    numeros=request.args
    if(numeros):
        print(numeros)
        signo_desigualdad = [int(x) for x in numeros.getlist('restricciones')]
        print(signo_desigualdad)
        signo_objetivo = [int(x) for x in numeros.getlist('objetivo')]
        print(signo_objetivo)
        signo_nuevos = [int(x) for x in numeros.getlist('nuevas')]
        restricciones=len(signo_desigualdad)
        variables=len(signo_nuevos)
        coeficientes={}
        for i in range(restricciones+1):
            coeficientes["key%s" %i] = [int(x) for x in numeros.getlist(str(i))]
        print(coeficientes)
        rows=restricciones+1
        columns=variables+1
        matriz=np.zeros((rows, columns))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)