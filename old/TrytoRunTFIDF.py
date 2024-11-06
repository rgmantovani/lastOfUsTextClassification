import pandas as pd
import numpy as np

# Convertendo o dicionário em um DataFrame
def transform_dict_to_df(dict_data):
    words = []
    vectors = []

    for key, value in dict_data.items():
        word_list_str = value[0]
        vector_str = value[1]

        # converter as strings para listas e numpy
        words_list = np.array(eval(word_list_str))
        vector_array = np.array(eval(vector_str)).flatten()

        words.append(words_list)
        vectors.append(vector_array)

    # Criar um DataFrame a partir dos vetores
    df = pd.DataFrame(vectors, columns=words[0])  # Usar a primeira lista de palavras como cabeçalho
    return df

dict_data = pd.read_csv('H:/IC_2023_2024/resultados_tfidf.csv')
# Aplicar a função ao dicionário
df_bow = transform_dict_to_df(dict_data)

# Exibir o DataFrame
print(df_bow)