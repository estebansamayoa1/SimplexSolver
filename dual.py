#Se importan las librerías a utilizar
import numpy as np 
from fractions import Fraction
import sympy as sp

"""# Maximización Dual"""

bM = -100000000000

def signosDmax(m, row, variables, matriz, constraints,rows):
  '''Aqui se llenan las variables de holgura, exceso y artificiales. Se le asigna M en la función objetivo a las variables artificiales.'''
  if m==1:
    matriz[row][constraints+row]=1
  if m==2:
    matriz[row][constraints+row]=-1
    matriz[row][constraints+row+variables]=1
    matriz[rows-1][constraints+row+variables]=bM
  if m==3:
    matriz[row][constraints+row]=0
    matriz[row][constraints+row+variables]=1
    matriz[rows-1][constraints+row+variables]=bM

  return matriz


def funcion_objetivoDmax(matriz,variables,rows,columns):
  '''Sobre la misma matriz anterior se agregan los coeficientes de las variables en la función objetivo que se busca maximizar'''
  for i in range(variables):
    val=int(input(f'Ingrese el valor de la variable {i+1} en la función objetivo\n'))
    matriz[rows-1,i]=val

  #print(matriz)
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print()
  print()
  
  return matriz,variables,rows,columns


def crear_matrizDmax():
  '''Aqui se define la función que crea la matriz, y se llena con los coeficientes de las restricciones'''
  constraints=int(input('¿Cuantas restricciones tiene su función?'))
  variables=int(input('¿Cuantas variables tiene su función?'))
  rows=constraints+1
  columns=variables+1
  matriz=np.zeros((rows, columns))
  sign=[]

  #Se hace un for para ir llenando las casillas con los valores de las restricciones
  for i in range(rows-1):
   for j in range(variables):
      val=float(input(f'Ingrese el valor de la variable {j+1} en la {i+1} restricción\n'))
      matriz[i][j]=val
  for i in range(rows-1):
      val=int(input(f'Ingrese el valor del resultado en la {i+1} restricción\n'))
      matriz[i,columns-1]=val*-1

  #Aqui se toman las desigualdades, se comprueba que los valores del lado derecho no sean negativos. Si lo son, multiplica la fila por -1 y cambia desigualdad.
  for i in range(0,variables):
    m=int(input(f'Signo de la variable{i}\n1.≤\n2.≥\n3.=\n'))
    sign.append(m)
    print(f'Original: {sign}')

  # se ingresan los valores de la funcion objetivo
  matriz,variables,rows,columns=funcion_objetivoDmax(matriz,variables,rows,columns)

  # Aqui se le hace transpose a la matriz con restricciones, objetivo y lado derecho
  matriz=matriz.transpose()
  # al cambio de la matriz cambia la cantidad de columnas
  a=rows
  rows=columns
  columns=a

  # se agregan los espacios de la matriz para variables de holgura y exceso
  cols_agregar=variables-1
  for i in range(cols_agregar):
      new_column = np.zeros((rows,1))
      matriz = np.insert(matriz, -1, new_column, axis=1)
  
  # array con las columnas ya agregadas
  columns=len(matriz[rows-1])
  print(f'NUEVAS COLUMNAS {columns}')

  print("ANTES DEL TRANSPOSE")
  #print(matriz)
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print() 
  print() 

  #Aqui se le asignan los signos finales de la desigualdad y se le agregan las variables. 
  for i in range(0,rows-1):
    matriz=signosDmax(sign[i], i, variables, matriz, constraints,rows)

  print("DESPUES DEL TRANSPOSE")
  #print(matriz)
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print() 
  print() 

  # cambia los signos segun su desigualdad
  for i in range(rows-1):
    if matriz[i][columns-1] < 0:
      for j in range(columns):
        matriz[i][j] *= -1
      if sign[i]==1:
        sign[i]=2
      elif sign[i]==2:
        sign[i]=1 
      print(sign)

  return matriz,variables,rows,columns


def eliminacion_mDmax(columns, matriz, rows):
  print(f'LA CANTIDAD DE COLUMNAS QUE USAR PARA ELIMINAR M:{columns}')
  ''' Aqui estamos eliminando las M, multiplicando la fila de funcion objetivo con la misma operacion de eliminacion''' 
  for h in range(0,columns):
      if matriz[rows-1][h]== bM:
        for i in range(rows):
          if matriz[i][h]==1:
            rowu=i
        print(f'row a user en la eliminacion M:{rowu}')
        for j in range(columns):
          if matriz[rowu][j] != 0:
            matriz[rows-1][j] = matriz[rows-1][j] - bM * matriz[rowu][j]
          else:
             matriz[rows-1][j] = matriz[rows-1][j]
  
  return matriz


def count_positivesDmax(matriz, columns,rows,pos):
  '''Esta función cuenta la cantidad de positivos que están en la ultima fila de la matriz'''
  pos=0
  for j in range(columns-1):
      if matriz[rows-1, j]>0:
        pos+=1
      else:
        continue
  return pos


def pivot_colDmax(matriz, columns,rows, col_pivote):
  '''Esta función busca la columna pivote, y el valor más positivo de la matriz para trabajar sobre él. Ignora la ultima columna de la fila objetivo.'''
  mas_pos=0
  for j in range(columns-1):
    if matriz[rows-1, j]>0:
      if matriz[rows-1, j]> mas_pos:
        mas_pos=matriz[rows-1, j] 
        col_pivote=j
  print(f'Columna: {col_pivote}')
  
  return col_pivote


def pivot_rDmax(matriz, columns,rows, col_pivote, pivot_val, pivot_row):
  '''Aqui se busca la row pivote, y el valor pivote para poder ir transformando la matriz.'''
  pivot_val=1000000000000000000000000000000
  for i in range(rows-1):
      val=matriz[i,columns-1]/matriz[i,col_pivote]
      # val=abs(val)
      if val < pivot_val:
        if val>0:
          pivot_val=val
          pivot_row=i
          print(pivot_val)
  print(f'El valor pivote es:{pivot_val}')
  print(f'Row: {pivot_row}')
  
  return pivot_val, pivot_row


def minimizarDmax():
  '''Aqui se hacen las iteraciones para encontrar el optimo del modelo'''
  matriz,variables,rows,columns=crear_matrizDmax()
  matriz=eliminacion_mDmax(columns, matriz, rows)
  print("inicial:")
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print()
  print()
  mas_pos=0
  pivot_row=0
  pivot_val=100
  col_pivote=0
  pos=1
  x=1

  #Se calcula la cantidad de positivos y se itera respecto a eso
  while(x==1):
    pos=count_positivesDmax(matriz,columns,rows,pos)
    print(f'Cantidad de positivos:{pos}')
    if pos>0:
      col_pivote=pivot_colDmax(matriz,columns,rows, col_pivote)
      pivot_val, pivot_row=pivot_rDmax(matriz,columns,rows,col_pivote, pivot_val, pivot_row)
      pos-=1
      nums=np.array([])
      for s in range(0,columns):
        a=matriz[pivot_row][s]/matriz[pivot_row][col_pivote]
        nums=np.append(nums, a)
      for s in range(columns):
        matriz[pivot_row][s]=nums[s]
      nums=np.array([])
      for i in range(rows):
        if i != pivot_row:
          val=matriz[i][col_pivote]
          for h in range(0,columns):
            a=matriz[i][h]- (val*matriz[pivot_row][h])
            matriz[i][h]=a

      #print(matriz)
      for row in matriz:
          for number in row:
            # se pone en fraccion los numeros y se le agrega style
              print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
          print()
      print()
    if pos==0:
      x+=1

  #Se hace la misma iteración 3 veces para ver si realmente se encuentra en la forma máximizada (matricialmente se puede ver cuando no hay positivos en la ultima fila)
  while(x==2):
    pos=count_positivesDmax(matriz,columns,rows,pos)
    print(f'Cantidad de positivos:{pos}')
    if pos>0:
      col_pivote=pivot_colDmax(matriz,columns,rows, col_pivote)
      pivot_val, pivot_row=pivot_rDmax(matriz,columns,rows,col_pivote, pivot_val, pivot_row)
      pos-=1
      nums=np.array([])
      for s in range(0,columns):
        a=matriz[pivot_row][s]/matriz[pivot_row][col_pivote]
        nums=np.append(nums, a)
      for s in range(columns):
        matriz[pivot_row][s]=nums[s]
      nums=np.array([])
      for i in range(rows):
        if i != pivot_row:
          val=matriz[i][col_pivote]
          for h in range(0,columns):
            a=matriz[i][h]- (val*matriz[pivot_row][h])
            matriz[i][h]=a

      #print(matriz)
      for row in matriz:
          for number in row:
            # se pone en fraccion los numeros y se le agrega style
              print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
          print()
      print()
    if pos==0:
      x+=1

  #Se hace una tercera iteración para comprobar sobre la matriz modificada, normalmente esta regresará la matriz como esta
  while(x==3):
    pos=count_positivesDmax(matriz,columns,rows,pos)
    print(f'Cantidad de positivos:{pos}')
    if pos>0:
      col_pivote=pivot_colDmax(matriz,columns,rows, col_pivote)
      pivot_val, pivot_row=pivot_rDmax(matriz,columns,rows,col_pivote, pivot_val, pivot_row)
      pos-=1
      nums=np.array([])
      #Aqui se divide la fila pivote para obtener un 1 en el valor pivote
      for s in range(0,columns):
        a=matriz[pivot_row][s]/matriz[pivot_row][col_pivote]
        nums=np.append(nums, a)
      for s in range(columns):
        matriz[pivot_row][s]=nums[s]
      nums=np.array([])
      #Aqui se hacen 0 los valores que se encuentren arriba o abajo del valor pivote, para ir optimizando la función
      for i in range(rows):
        if i != pivot_row:
          val=matriz[i][col_pivote]
          for h in range(0,columns):
            a=matriz[i][h]- (val*matriz[pivot_row][h])
            matriz[i][h]=a

      #print(matriz)
      for row in matriz:
          for number in row:
            # se pone en fraccion los numeros y se le agrega style
              print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
          print()
      print()
      
    #Si llega a la cantidad de positivos que se calculo al principio, retorna el valor de las x y el valor minimizado de la función 
    if pos==0:
      counx=[]
      for j in range(variables):
        counts=[]
        for i in range(rows-1):
          if matriz[i][j]==1:
            counts.append(i)
        counx.append(counts)
      for h in range(variables):
        for i in range(0,rows-1):
          if len(counx[h])>1:
            print(f"El valor de x{h+1} es 0")
            break
          elif len(counx[h])==1:
            print(f"El valor de x{h+1} es ", Fraction(str(matriz[counx[h][0]][columns-1])).limit_denominator(100))
            break
      print('Valor Máximizado: ',Fraction(str(matriz[rows-1][columns-1])).limit_denominator(100))
      break


"""# Minimizacion Dual"""

M = 100000000000

def signosDmin(m, row, variables, matriz, constraints,rows):
  '''Aqui se llenan las variables de holgura, exceso, y artificiales segun la desigualdad. Se le asigna M a las variables artificiales.''' 
  if m==1:
    matriz[row, variables+row]=1
  if m==2:
    matriz[row][variables+row]=-1
    matriz[row][variables+row+constraints-1]=1
    matriz[rows-1][variables+row+constraints-1]=M
  if m==3:
    matriz[row][variables+row]=0
    matriz[row][variables+row+constraints]=1
    matriz[rows-1][variables+row+constraints]=M
  
  return matriz


def funcion_objetivoDmin(matriz,variables,rows,columns):
  '''Sobre la misma matriz anterior se agregan los coeficientes de las variables en la función objetivo que se busca maximizar'''
  for i in range(variables):
    val=int(input(f'Ingrese el valor de la variable {i+1} en la función objetivo\n'))
    matriz[rows-1,i]=val

  #print(matriz)
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print()
  print()
  
  return matriz,variables,rows,columns


def crear_matrizDmin():
  '''Aqui se define la función que crea la matriz, y se llena con los coeficientes de las restricciones'''
  constraints=int(input('¿Cuantas restricciones tiene su función?'))
  variables=int(input('¿Cuantas variables tiene su función?'))
  rows=constraints+1
  columns=variables+1
  matriz=np.zeros((rows, columns))
  sign=[]

  #Se hace un for para ir llenando las casillas con los valores de las restricciones
  for i in range(rows-1):
   for j in range(variables):
      val=float(input(f'Ingrese el valor de la variable {j+1} en la {i+1} restricción\n'))
      matriz[i][j]=val
  for i in range(rows-1):
      val=int(input(f'Ingrese el valor del resultado en la {i+1} restricción\n'))
      matriz[i,columns-1]=val*-1

  #Aqui se toman las desigualdades, se comprueba que los valores del lado derecho no sean negativos. Si lo son, multiplica la fila por -1 y cambia desigualdad.
  for i in range(0,variables):
    m=int(input(f'Signo de la variable{i}\n1.≤\n2.≥\n3.=\n'))
    sign.append(m)
    print(f'Original: {sign}')

  # se le agregan los valores a la funcion objetivo
  matriz,variables,rows,columns=funcion_objetivoDmin(matriz,variables,rows,columns)

  # se le hace transpose a la matriz con todos los valores
  matriz=matriz.transpose()
  # se identifica la nueva cantidad de columnas que tiene la matriz ya transpuesta
  a=rows
  rows=columns
  columns=a

  # se le agrega a la matriz los espacios para las variables de holgura y exceso
  cols_agregar=variables-1
  for i in range(cols_agregar):
      new_column = np.zeros((rows,1))
      matriz = np.insert(matriz, -1, new_column, axis=1)
  
  columns=len(matriz[rows-1])
  print(f'NUEVAS COLUMNAS {columns}')


  print("ANTES DEL TRANSPOSE")
  #print(matriz)
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print() 
  print() 

  #Aqui se le asignan los signos finales de la desigualdad y se le agregan las variables. 
  for i in range(0,rows-1):
    matriz=signosDmin(sign[i], i, variables, matriz, constraints,rows)

  print("DESPUES DEL TRANSPOSE")
  #print(matriz)
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print() 
  print() 

  # cambio de signo segun la desigualdad de la restriccion
  for i in range(rows-1):
    if matriz[i][columns-1] < 0:
      for j in range(columns):
        matriz[i][j] *= -1
      if sign[i]==1:
        sign[i]=2
      elif sign[i]==2:
        sign[i]=1 
      print(sign)

  return matriz,variables,rows,columns


def eliminacion_mDmin(columns, matriz, rows):
  ''' Aqui estamos eliminando las M, multiplicando la fila de funcion objetivo con la misma operacion de eliminacion''' 
  for h in range(0,columns):
      if matriz[rows-1][h]== M:
        for i in range(rows):
          if matriz[i][h]==1:
            rowu=i
        print(f'row a user en la eliminacion M:{rowu}')
        for j in range(columns):
          if matriz[rowu][j] != 0:
            matriz[rows-1][j] = matriz[rows-1][j] - M * matriz[rowu][j]
          else:
             matriz[rows-1][j] = matriz[rows-1][j]
  
  return matriz


def pivot_colDmin(matriz, columns,rows, col_pivote):
  '''Esta función busca la columna pivote, y el valor más negativo de la matriz para trabajar sobre él'''
  mas_neg=0
  for j in range(columns-1):
    if matriz[rows-1, j]<0:
      if matriz[rows-1, j]< mas_neg:
        mas_neg=matriz[rows-1, j] 
        col_pivote=j
  print(f'Columna: {col_pivote}')
  
  return col_pivote


def count_negativesDmin(matriz, columns,rows,negs):
  '''Esta función cuenta la cantidad de negativos que están en la ultima fila de la matriz '''
  negs=0
  for j in range(columns-1):
      if matriz[rows-1, j]<0:
        negs+=1
      else:
        continue
  
  return negs


def pivot_rDmin(matriz, columns,rows, col_pivote, pivot_val, pivot_row):
  '''Aqui se busca la row pivote, y el valor pivote para poder ir transformando la matriz.'''
  pivot_val=1000000000000000000000000000000
  for i in range(rows-1):
    denominador = matriz[i,col_pivote]
    if denominador == 0:
      denominador = 1
    val=matriz[i,columns-1]/denominador
    # val=abs(val)
    if val < pivot_val:
      if val>0:
        pivot_val=val
        pivot_row=i
        print(pivot_val)
  print(f'El valor pivote es:{pivot_val}')
  print(f'Row: {pivot_row}')
  
  return pivot_val, pivot_row


def maximizarDmin():
  '''Aqui se realizan las iteraciones para encontrar la maximización del modelo'''
  matriz,variables,rows,columns=crear_matrizDmin()
  matriz=eliminacion_mDmin(columns, matriz, rows)
  print("inicial:")
  for row in matriz:
      for number in row:
        # se pone en fraccion los numeros y se le agrega style
          print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
      print()
  print()
  mas_neg=0
  pivot_row=0
  pivot_val=100
  col_pivote=0
  negs=1
  x=1

  #Se calcula la cantidad de negativos y se itera respecto a eso
  while(x==1):
    negs=count_negativesDmin(matriz,columns,rows,negs)
    print(f'Cantidad de negativos:{negs}')
    if negs>0:
      col_pivote=pivot_colDmin(matriz,columns,rows, col_pivote)
      pivot_val, pivot_row=pivot_rDmin(matriz,columns,rows,col_pivote, pivot_val, pivot_row)
      negs-=1
      nums=np.array([])
      for s in range(0,columns):
        a=matriz[pivot_row][s]/matriz[pivot_row][col_pivote]
        nums=np.append(nums, a)
      for s in range(columns):
        matriz[pivot_row][s]=nums[s]
      nums=np.array([])
      for i in range(rows):
        if i != pivot_row:
          val=matriz[i][col_pivote]
          for h in range(0,columns):
            a=matriz[i][h]- (val*matriz[pivot_row][h])
            matriz[i][h]=a

      #print(matriz)
      for row in matriz:
          for number in row:
            # se pone en fraccion los numeros y se le agrega style
              print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
          print()
      print()
    if negs==0:
      x+=1

  #Se hace la misma iteración 3 veces para ver si realmente se encuentra en la forma máximizada (matricialmente se puede ver cuando no hay negativos en la ultima fila)
  while(x==2):
    negs=count_negativesDmin(matriz,columns,rows,negs)
    print(f'Cantidad de negativos:{negs}')
    if negs>0:
      col_pivote=pivot_colDmin(matriz,columns,rows, col_pivote)
      pivot_val, pivot_row=pivot_rDmin(matriz,columns,rows,col_pivote, pivot_val, pivot_row)
      negs-=1
      nums=np.array([])
      for s in range(0,columns):
        a=matriz[pivot_row][s]/matriz[pivot_row][col_pivote]
        nums=np.append(nums, a)
      for s in range(columns):
        matriz[pivot_row][s]=nums[s]
      nums=np.array([])
      for i in range(rows):
        if i != pivot_row:
          val=matriz[i][col_pivote]
          for h in range(0,columns):
            a=matriz[i][h]- (val*matriz[pivot_row][h])
            matriz[i][h]=a

      #print(matriz)
      for row in matriz:
          for number in row:
            # se pone en fraccion los numeros y se le agrega style
              print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
          print()
      print()
    if negs==0:
      x+=1

  #Se hace una tercera iteración para comprobar sobre la matriz modificada, normalmente esta regresará la matriz como esta
  while(x==3):
    negs=count_negativesDmin(matriz,columns,rows,negs)
    print(f'Cantidad de negativos:{negs}')
    if negs>0:
      col_pivote=pivot_colDmin(matriz,columns,rows, col_pivote)
      pivot_val, pivot_row=pivot_rDmin(matriz,columns,rows,col_pivote, pivot_val, pivot_row)
      negs-=1
      nums=np.array([])
      #Aqui se divide la fila pivote para obtener un 1 en el valor pivote
      for s in range(0,columns):
        a=matriz[pivot_row][s]/matriz[pivot_row][col_pivote]
        nums=np.append(nums, a)
      for s in range(columns):
        matriz[pivot_row][s]=nums[s]
      nums=np.array([])
      #Aqui se hacen 0 los valores que se encuentren arriba o abajo del valor pivote, para ir optimizando la función
      for i in range(rows):
        if i != pivot_row:
          val=matriz[i][col_pivote]
          for h in range(0,columns):
            a=matriz[i][h]- (val*matriz[pivot_row][h])
            matriz[i][h]=a

      #print(matriz)
      for row in matriz:
          for number in row:
            # se pone en fraccion los numeros y se le agrega style
              print(Fraction(str(number)).limit_denominator(100), end ='|\t') 
          print()
      print()
      
    #Si llega a la cantidad de negativos que se calculo al principio, retorna el valor de las x y el valor maximizado de la función 
    if negs==0:
      counx=[]
      for j in range(variables):
        counts=[]
        for i in range(rows-1):
          if matriz[i][j]==1:
            counts.append(i)
        counx.append(counts)
      for h in range(variables):
        for i in range(0,rows-1):
          if len(counx[h])>1:
            print(f"El valor de x{h+1} es 0")
            break
          elif len(counx[h])==1:
            print(f"El valor de x{h+1} es ", Fraction(str(matriz[counx[h][0]][columns-1])).limit_denominator(100))
            break
      print('Valor Minimizado: ',Fraction(str(matriz[rows-1][columns-1])).limit_denominator(100))
      break


"""# Main"""

def main():
  '''Una función main para que el usuario decida si quiere realizar un problema de maximización o de minimización'''
  minomax=int(input('¿Que desea realizar?\n1.Maximizar\n2.Minimizar\n'))
  if minomax==1:
    minimizarDmax()
  if minomax==2:
    maximizarDmin()

main()