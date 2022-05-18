# SimplexSolver
##### Esteban Samayoa, 20200188
##### Marcela Melgar, 20200487

## Problemática a Resolver
Uno de los problemas principales que las empresas enfrentan actualmente es el de la toma de decisiones sobre cuántos productos manufacturar respecto a los insumos que poseen, buscando siempre maximizar su utilidad o ya bien, minimizar sus gastos. Por lo que es importante que cuenten con las herramientas adecuadas y confiables para poder realizar dichos cálculos.


## Propuesta de Proyecto
Nuestra solución a dicho problema es realizar una página web en la cual el usuario puede ingresar de manera fácil y rápida sus restricciones (insumos, mano de obra, etc.) por cada producto que la empresa realiza, y además sus costos y precios de venta. El programa realizará de manera automática el cálculo necesario para determinar cuál es la combinación óptima de productos para maximizar sus ganancias o reducir sus costos. 

El sitio web será interactivo en el sentido que el usuario puede cambiar la cantidad de restricciones (insumos) que posee y la cantidad de productos que realiza para satisfacer sus necesidades específicas. 

Consideramos que este proyecto no solo les permitirá realizar estos cálculos de manera rápida, sino que también les ayudará a tomar decisiones sobre cuánta materia prima adquirir, cuanta mano de obra necesitará y cuánto podrán producir en total, además que podrán visualizar cuánto es su ganancia maximizada o su costo minimizado, por lo que consideramos que es una herramienta muy útil para muchas diferentes industrias manufactureras. 


## Entregables Propuestos
**Página web funcionando:** La página web con diseño amigable al usuario donde podrá ingresar sus restricciones y sus variables de manera sencilla e interactiva, y además podrá escoger el método con el cual realizar sus cálculos (ya sea Big M ó Dual Simplex).

**Códigos de Backend que realicen los cálculos automáticos:** Los códigos que realizan los cálculos automáticos para encontrar los valores maximizados de cada sistema de programación lineal y así ofrecer los valores finales al cliente. 

**Tabla de resultados finales con los valores maximizados y/o minimizados:** Dentro de la página web, los usuarios recibirán los valores finales en una tabla simple donde los podrán evaluar y en todo caso realizar otra vez el cálculo ya sea cambiando el método (aunque realmente no hay diferencia en los valores) o ya bien cambiando sus variables y restricciones para ir encontrando la mejor combinación. Estos valores les permitirán tomar decisiones a las empresas para ver cómo maximizar sus ganancias.

**Ejemplo de Implementación:** Aparte, adjuntamos un problema de insumos aplicado a la vida real, con los pasos para que se familiaricen con la plataforma.

## Herramientas
Para la elaboración del proyecto utilizamos distintos lenguajes, librerías y herramientas para que se pudiera llevar a cabo con éxito la página web.
- **BackEnd:** Python con las librerías de Flask para levantar la página web, Numpy para arrays y operaciones, Fractions para mejor impresión de números no enteros.
- **FrontEnd:** HTML, en la que utilizamos un template de W3schools para dimensiones y bootstrap.
- **Interactividad:** JavaScript en la cual se toman las variables que ingresa el usuario para poderlas utilizar en las funciones de Python.
