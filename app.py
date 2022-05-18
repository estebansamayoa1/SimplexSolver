from flask import Flask, render_template, request, redirect, url_for
import numpy as np 
from fractions import Fraction
import sympy as sp
from dual import minimizarDmax, maximizarDmin
import ast



app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def solver():
    numeros=request.args
    if(numeros):
        print(numeros)
        signo_desigualdad = [int(x) for x in numeros.getlist('restricciones')]
        print(signo_desigualdad)
        signo_nuevos = [int(x) for x in numeros.getlist('nuevas')]
        metodo=[str(x) for x in numeros.getlist('metodo')]
        print(metodo)
        print(signo_nuevos)
        restricciones=len(signo_desigualdad)
        variables=len(signo_nuevos)
        coeficientes={}
        for i in range(restricciones+1):
            coeficientes["key%s" %i] = [int(x) for x in numeros.getlist(str(i))]
        arreglox=[]
        for constraint in coeficientes:
            arreglox.append(coeficientes[constraint]) 
        print(f"METODO:{metodo}")
        info={
            "infomatriz":arreglox,
            "metodo":metodo[0],
            "signos":signo_nuevos,
            "variables":variables,
            "restricciones":restricciones
        }
        return redirect(url_for('respuesta', data=info))
    return render_template('index.html')

@app.route('/respuesta', methods=["GET", "POST"])
def respuesta():
    data=request.args['data']
    data=ast.literal_eval(data)
    infomatriz=data['infomatriz']
    metodo=data['metodo']
    print(metodo)
    signos=data['signos']
    variables=data['variables']
    restricciones=data['restricciones']
    matriz=np.array(infomatriz) 
    rows=restricciones+1
    columns=variables+1
    matrizfinal=np.zeros((rows,columns))
    if metodo=="Maximizar":
            matrizfinal,val,soluciones=minimizarDmax(matriz, variables, restricciones, columns, rows, signos)
    elif metodo=="Minimizar":
            matrizfinal,val,soluciones=maximizarDmin(matriz, variables, restricciones, columns, rows, signos)
    return render_template("respuesta.html", matrizfinal=matrizfinal, val=val, soluciones=soluciones)

if __name__ == "__main__":
    app.run(port=8000,debug=True)