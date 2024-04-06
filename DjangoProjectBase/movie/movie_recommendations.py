from dotenv import load_dotenv, find_dotenv
import json
import os
from openai import OpenAI
#from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

with open('D:\\U\\5 Semestre\\Proyecto int\\Taller 3\\Taller3-PI1\\DjangoProjectBase\\movie\\movie_descriptions_embeddings.json', 'r') as file:

    file_content = file.read()
    movies = json.loads(file_content)

#Esta función devuelve una representación numérica (embedding) de un texto, en este caso
#la descripción de las películas
    
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   load_dotenv()
   OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

   client = OpenAI(api_key=OPENAI_API_KEY)
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#Si se tuviera un prompt por ejemplo: Película de la segunda guerra mundial, podríamos generar el embedding del prompt y comparar contra 
#los embeddings de cada una de las películas de la base de datos. La película con la similitud más alta al prompt sería la película
#recomendada.

req = "Accion"
emb = get_embedding(req)

sim = []
for i in range(len(movies)):
  sim.append(cosine_similarity(emb,movies[i]['embedding']))
sim = np.array(sim)
idx = np.argmax(sim)
print(movies[idx]['title'])
"""
sim = []
for i in range(len(movies)):
    sim.append((movies[i]['title'], cosine_similarity(emb, movies[i]['embedding'])))

# Ordenar las películas por similitud descendente
sim_sorted = sorted(sim, key=lambda x: x[1], reverse=True)

# Seleccionar las mejores recomendaciones (por ejemplo, las tres más similares)
num_recommendations = 3
recommendations = sim_sorted[:num_recommendations]

# Imprimir las recomendaciones
print("Recomendaciones:")
for recommendation in recommendations:
    print(recommendation[0])
"""
