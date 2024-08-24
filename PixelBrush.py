import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import imageio

# Configuración de parámetros
num_individuals = 50
num_generations = 500
num_triangles = 50
mutation_rate = 0.01
elite_percentage = 0.1

# Crear la carpeta 'img' si no existe
if not os.path.exists('img'):
    os.makedirs('img')

# Preprocesar la imagen de entrada (sin necesidad de convertir a escala de grises)
def preprocess_image(image_path):
    image = Image.open(image_path)
    return np.array(image) / 255.0

# Crear un individuo
def create_individual(image_size):
    return [(random.randint(0, image_size[0]), random.randint(0, image_size[1]), 
             random.randint(0, image_size[0]), random.randint(0, image_size[1]), 
             random.randint(0, image_size[0]), random.randint(0, image_size[1]), 
             random.uniform(0, 1)) for _ in range(num_triangles)]

# Dibujar un individuo en una imagen
def draw_individual(individual, image_size):
    image = Image.new('L', image_size, color=255)
    draw = ImageDraw.Draw(image, 'L')
    for triangle in individual:
        draw.polygon([triangle[0:2], triangle[2:4], triangle[4:6]], fill=int(triangle[6]*255))
    return np.array(image) / 255.0

# Función de fitness
def fitness(individual, target_image):
    generated_image = draw_individual(individual, target_image.shape)
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

# Mutación
def mutate(individual, image_size):
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
    image = Image.fromarray((draw_individual(individual, image_size) * 255).astype(np.uint8))
    image.save(os.path.join('img', filename))

# Algoritmo genético principal
def genetic_algorithm(target_image, image_size):
    population = [create_individual(image_size) for _ in range(num_individuals)]
    best_fitness = []
    average_fitness = []
    best_images = []

    for generation in range(num_generations):
        fitness_scores = [fitness(ind, target_image) for ind in population]
        best_fitness.append(max(fitness_scores))
        average_fitness.append(np.mean(fitness_scores))
        best_individual = population[np.argmax(fitness_scores)]

        save_image(best_individual, image_size, f"generation_{generation}.jpg")
        best_images.append(os.path.join('img', f"generation_{generation}.jpg"))

        selected_population = select_population(population, fitness_scores)
        new_population = []

        while len(new_population) < num_individuals:
            parent1, parent2 = random.sample(selected_population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, image_size)
            new_population.append(child)

        population = new_population

    # Mostrar gráficos de la evolución del fitness
    plt.plot(range(num_generations), best_fitness, label="Best Fitness")
    plt.plot(range(num_generations), average_fitness, label="Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    # Crear un GIF con la evolución de la imagen
    images = [imageio.imread(image) for image in best_images]
    imageio.mimsave(os.path.join('img', "evolution.gif"), images, duration=0.2)

# Ejecución del algoritmo
image_path = 'Prueba.jpeg'
target_image = preprocess_image(image_path)
image_size = target_image.shape
genetic_algorithm(target_image, image_size)
