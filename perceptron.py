import numpy as np
print(np.__version__)  # deve imprimir a versão sem erro

# Variaveis de entrada 
entradas = np.array([ -1, 7, 5])

# Definir as variáveis de peso iniciais, números aleatórios = w
pesos = np.array([0.8, 0.1, 0])

# Criar a funcção soma 
def somaFunction(entradas, pesos):
    # Produto Escalar (multiplicar e somar)
    # É muito mais rapido do que o for acima, e é a forma como as bibliotecas de machine learning fazem
    return entradas.dot(pesos)  # multiplicação de matrizes

# Executar a função soma passando entradas e pesos
soma = somaFunction(entradas, pesos)
print("retorno = ", soma)

# Criar a funcção de ativação - Step Function
def stepFunction(soma):
   if (soma >= 1):
      return 1
   return 0 

# Executando a função de ativação para saber se o neurônio dispara ou não 
resultado = stepFunction(soma)
print("resultado = ", resultado)