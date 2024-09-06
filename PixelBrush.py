import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import imageio

# Configuración de parámetros
num_individuals = 50
num_generations = 500  # Se usará solo si no se establece fitness_threshold
num_triangles = 50
initial_mutation_rate = 0.01  # Tasa de mutación inicial
elite_percentage = 0.05  # Elitismo controlado: 5% de los mejores individuos
new_individuals_rate = 0.1  # Introducción de nuevos individuos en cada generación
fitness_threshold = None  # Establecer este valor para detener cuando se alcance

# Crear la carpeta 'img' si no existe
if not os.path.exists('img'):
    os.makedirs('img')

# Preprocesar la imagen de entrada (convertir a escala de grises)
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    bw_image_name = os.path.splitext(image_path)[0] + "BW.jpg"
    image.save(bw_image_name)  # Guardar la imagen en blanco y negro
    return np.array(image) / 255.0

# Crear un individuo
def create_individual(image_size):
    return [(random.randint(0, image_size[0]), random.randint(0, image_size[1]), 
             random.randint(0, image_size[0]), random.randint(0, image_size[1]), 
             random.randint(0, image_size[0]), random.randint(0, image_size[1]), 
             random.uniform(0, 1)) for _ in range(num_triangles)]

# Dibujar un individuo en una imagen
def draw_individual(individual, image_size):
    image = Image.new('L', image_size[:2], color=255)  # Solo usar los dos primeros elementos de image_size
    draw = ImageDraw.Draw(image, 'L')
    for triangle in individual:
        draw.polygon([triangle[0:2], triangle[2:4], triangle[4:6]], fill=int(triangle[6]*255))
    return np.array(image) / 255.0

# Función de fitness
def fitness(individual, target_image):
    generated_image = draw_individual(individual, target_image.shape[:2])  # Solo usar los dos primeros elementos de shape
    return -np.sum(np.abs(generated_image - target_image))

# Selección
def select_population(population, fitness_scores):
    elite_size = int(len(population) * elite_percentage)
    selected = np.argsort(fitness_scores)[-elite_size:]
    return [population[i] for i in selected]

# Cruce
def crossover(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        if random.random() > 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

# Mutación adaptativa
def mutate(individual, image_size, mutation_rate):
    for i in range(len(individual)):
        triangle = list(individual[i])  # Convertir a lista para mutar
        if random.random() < mutation_rate:
            index = random.randint(0, 6)
            if index < 6:
                triangle[index] = random.randint(0, image_size[index % 2])
            else:
                triangle[index] = random.uniform(0, 1)
        individual[i] = tuple(triangle)  # Volver a convertir en tupla
    return individual

# Guardar la mejor imagen de cada generación
def save_image(individual, image_size, filename):
    image = Image.fromarray((draw_individual(individual, image_size[:2]) * 255).astype(np.uint8))
    image.save(os.path.join('img', filename))

# Algoritmo genético principal
def genetic_algorithm(target_image, image_size, fitness_threshold=None):
    population = [create_individual(image_size) for _ in range(num_individuals)]
    best_fitness = []
    average_fitness = []
    best_images = []
    mutation_rate = initial_mutation_rate

    generation = 0
    while True:
        fitness_scores = [fitness(ind, target_image) for ind in population]
        best_fitness_score = max(fitness_scores)
        best_fitness.append(best_fitness_score)
        average_fitness.append(np.mean(fitness_scores))
        best_individual = population[np.argmax(fitness_scores)]

        save_image(best_individual, image_size, f"generation_{generation}.jpg")
        best_images.append(os.path.join('img', f"generation_{generation}.jpg"))

        if fitness_threshold and best_fitness_score >= fitness_threshold:
            print(f"Fitness threshold of {fitness_threshold} reached at generation {generation}.")
            break

        if generation >= num_generations and not fitness_threshold:
            break

        selected_population = select_population(population, fitness_scores)
        new_population = []

        # Crear nueva población a través de cruces y mutaciones
        while len(new_population) < num_individuals * (1 - new_individuals_rate):
            parent1, parent2 = random.sample(selected_population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, image_size, mutation_rate)
            new_population.append(child)

        # Introducir nuevos individuos aleatorios
        while len(new_population) < num_individuals:
            new_population.append(create_individual(image_size))

        population = new_population
        generation += 1

        # Ajustar la tasa de mutación adaptativamente
        diversity = np.std(fitness_scores)
        if diversity < 0.1:  # Umbral de diversidad 
            mutation_rate = min(1.0, mutation_rate * 1.5)
        else:
            mutation_rate = initial_mutation_rate

    # Mostrar gráficos de la evolución del fitness
    plt.plot(range(generation + 1), best_fitness, label="Best Fitness")
    plt.plot(range(generation + 1), average_fitness, label="Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    # Crear un GIF con la evolución de la imagen
    images = [imageio.imread(image) for image in best_images]
    imageio.mimsave(os.path.join('img', "evolution.gif"), images, duration=0.2)

# Ejecución del algoritmo
image_path = 'Original.jpg'
target_image = preprocess_image(image_path)
image_size = target_image.shape
genetic_algorithm(target_image, image_size, fitness_threshold=-50)
