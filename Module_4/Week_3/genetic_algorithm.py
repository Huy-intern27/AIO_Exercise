import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import random
random.seed(0)
np.random.seed(0)

def load_data_from_file(file_name = './data/advertising.csv'):
  data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  features_X = data[:, :3]
  sales_Y = data[:, 3]
  features_X = np.c_[np.ones((features_X.shape[0], 1)), features_X]

  return features_X, sales_Y

features_X, sales_Y = load_data_from_file()

def generate_random_value(bound = 10):
  return (random.random() - 0.5) * bound

def create_individual(n=4, bound=10):
  individual = []
  for _ in range(n):
    individual.append(generate_random_value(bound))

  return individual

def compute_loss(individual):
  theta = np.array(individual)
  y_hat = features_X.dot(theta)
  loss = np.multiply((y_hat - sales_Y), (y_hat - sales_Y)).mean()
  return loss

def compute_fitness(individual):
  loss = compute_loss(individual)
  fitness_value = 1 / (loss + 1)

  return fitness_value

def crossover(individual1, individual2, crossover_rate=0.9):
  individual1_new = individual1.copy()
  individual2_new = individual2.copy()

  for i in range (len(individual1)):
    if random.random() < crossover_rate:
      individual1_new[i] = individual2[i]
      individual2_new[i] = individual1[i]

  return individual1_new, individual2_new

def mutate(individual, mutation_rate=0.05):
  individual_new = individual.copy()
  for i in range(len(individual)):
    if random.random() < mutation_rate:
      individual_new[i] = generate_random_value()

  return individual_new

def initialize_population(m):
  population = [create_individual(n=4, bound=10) for _ in range(m)]
  return population

def selection(sorted_old_population, m=100):
  index1 = random.randint(0, m - 1)
  while True:
    index2 = random.randint(0, m - 1)
    if index1 != index2:
      break
  individual_s = sorted_old_population[max(index1, index2)]

  return individual_s

def create_new_population(old_population, elitism=2, gen=1):
  m = len(old_population)
  sorted_population = sorted(old_population, key=compute_fitness, reverse=True)

  if gen % 1 == 0:
    print("Best loss:", compute_loss(sorted_population[m - 1]), "with chromosome:", sorted_population[m - 1])
  new_population = sorted_population[:elitism]
  while len(new_population) < m:
    parent1 = selection(sorted_population)
    parent2 = selection(sorted_population)

    child1, child2 = crossover(parent1, parent2)
    child1 = mutate(child1)
    child2 = mutate(child2)

    new_population.append(child1)
    new_population.append(child2)
  new_population = new_population[:m]

  return new_population, compute_loss(sorted_population[m-1])

def run_ga():
  n_generations = 100
  m = 600
  features_X, sales_Y = load_data_from_file()
  population = initialize_population(m)
  losses_list = []
  for i in range(n_generations):
    population, loss = create_new_population(population, elitism=2, gen=i)
    losses_list.append(loss)
  return losses_list

def visualize_loss(losses_list):
  plt.plot(losses_list)
  plt.xlabel('Generation')
  plt.ylabel('Loss')
  plt.title('Loss vs Generation')
  plt.show()

losses_list = run_ga()
visualize_loss(losses_list)