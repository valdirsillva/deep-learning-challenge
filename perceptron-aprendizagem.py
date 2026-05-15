import numpy as np

# Definir matriz de entradas 
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Definir a Classe Verdade
saidas = np.array([0, 0, 0, 1])  # AND lógico
# Definir os pesos iniciais
pesos = np.array([0.0, 0.0])

# definira variavel de taxaAprendizagem
taxa_aprendizagem = 0.1

# Definir a função stepFunction 
def  stepFunction(soma):
   if (soma >= 1):
      return 1
   return 0

# Definir a função de soma
def somaFunction(entradas):
   #  Multiplicar os pesos
   return entradas.dot(pesos)

# Definindo a Função de treino do Modelo
def treinarModelo():
   erro_total = 1
   while erro_total != 0:
      erro_total = 0
      for i in range(len(saidas)):
         #Chamada da Função soma para aplicação dos pesos  
         soma = somaFunction(entradas[i])

         #Chamada da Função de ativação
         saidaCalculada = stepFunction(soma)

         # Calculo do erro
         erro = abs(saidas[i] - saidaCalculada)

        # Adicionar o resultado na variavel erro_total
         erro_total += erro

        #  Realizar os ajustes nos pesos
         for j in range(len(pesos)):
            # Formula de atualização dos pesos
            # peso(n + 1) = peso(n) + (taxa_aprendizagem * entrada * erro)
            pesos[j] = pesos[j] + (taxa_aprendizagem * entradas[i][j] * erro)
            # Peso atualizado
            print("Pesos atualizados: " + str(pesos[j]))
      print("Total de erros: " + str(erro_total))

# Executar o treino do Modelo
treinarModelo()
print('Rede Neural Treinada com Sucesso!')
print(stepFunction(somaFunction(entradas[0])))
print(stepFunction(somaFunction(entradas[1])))
print(stepFunction(somaFunction(entradas[2])))  
print(stepFunction(somaFunction(entradas[3])))