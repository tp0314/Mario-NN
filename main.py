from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import cv2
import random
from math import *
import time
import os
from EM_Ram import *
import multiprocessing
import copy

class Element:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.game = Game()
        self.obstacle_grid = None
        self.input = 0
        self.weights = (0.10 * np.random.randn(number_of_inputs, 18))
        self.biases = np.zeros((1,18))

        self.hidden_layer_weights = (0.10 * np.random.randn(18, number_of_neurons)) ##hidden layer of 18 neurons
        self.output_biases = np.zeros((1, number_of_neurons))
        
        self.fitness = 0

    def forward(self,inputs, weights, biases):
        self.output = np.dot(inputs, weights) + biases

    def activation_ReLU(self, inputs):
        self.output =  np.maximum(0, inputs) 
    
    def generate_game_input(self):
        self.input = np.argmax(self.output) 

    def measure_fitness(self):
        self.fitness += self.game.reward

    def breed(self, partner):
        child = Element(208,12)
        num_hidden = self.hidden_layer_weights.shape[0]

        for h in range(num_hidden):
            if random.choice([0,1]):
                source = self
            else:
                source = partner
            child.weights[:, h] = source.weights[:,h]
            child.biases[0,h] = source.biases[0,h]
            child.hidden_layer_weights[h,:] = source.hidden_layer_weights[h,:]

        for i in range(len(self.output_biases[0])):
            if random.choice([0,1]):
                source = self
            else:
                source = partner
            child.output_biases[0][i] = source.output_biases[0][i]
        return child

        # for i in range(len(self.weights)):
        #     for j in range(len(self.weights[i])):
        #         if random.choice([0, 1]):
        #             child.weights[i][j] = partner.weights[i][j]
        #         else:
        #             child.weights[i][j] = self.weights[i][j]

        # for i in range(len(self.biases)):
        #     for j in range(len(self.biases[i])):
        #         if random.choice([0, 1]):
        #             child.biases[i][j] = partner.biases[i][j]
        #         else:
        #             child.biases[i][j] = self.biases[i][j]

        # for i in range(len(self.hidden_layer_weights)):
        #     for j in range(len(self.hidden_layer_weights[i])):
        #         if random.choice([0, 1]):
        #             child.hidden_layer_weights[i][j] = partner.hidden_layer_weights[i][j]
        #         else:
        #             child.hidden_layer_weights[i][j] = self.hidden_layer_weights[i][j]

        # for i in range(len(self.output_biases)):
        #     for j in range(len(self.output_biases[i])):
        #         if random.choice([0, 1]):
        #             child.output_biases[i][j] = partner.output_biases[i][j]
        #         else:
        #             child.output_biases[i][j] = self.output_biases[i][j]
        # return child

def create_population(n):
    elements = []
    for _ in range(n):
        elements.append(Element(208, 12)) 
    return elements

def clone_network(e):
    child = Element(208,12)
    child.weights = e.weights.copy()
    child.biases = e.biases.copy()
    child.hidden_layer_weights = e.hidden_layer_weights.copy()
    child.output_biases = e.output_biases.copy()
    return child

def generate_mating_pool():
    mating_pool = []
    for e in elements:
        n = floor(e.fitness / 10)
        for _ in range(n):
            mating_pool.append(e)

    print(len(mating_pool))
    return mating_pool

def breed_new_pop(n, mating_pool, population):
    n_of_elites = max(2,int(n*.15))

    elites = sorted(population, key = lambda e: e.fitness, reverse = True)[:n_of_elites]
    new_population = [clone_network(e) for e in elites]

    for _ in range(n - n_of_elites):
        parent_a = random.choice(mating_pool)
        parent_b = random.choice(mating_pool)
        child = parent_a.breed(parent_b)
        new_population.append(child)
    
    return new_population, n_of_elites

def mutate_population(population, n_of_elites):
    for e in population[n_of_elites:]:
        for i in range(len(e.weights)):
            for j in range(len(e.weights[i])):
                if random.random() < 0.1:
                    e.weights[i][j] += random.uniform(-0.08, 0.08)

        for i in range(len(e.biases)):
            for j in range(len(e.biases[i])):
                if random.random() < 0.1:
                    e.biases[i][j] += random.uniform(-0.08, 0.08)

        for i in range(len(e.hidden_layer_weights)):
            for j in range(len(e.hidden_layer_weights[i])):
                if random.random() < 0.1:
                    e.hidden_layer_weights[i][j] += random.uniform(-0.05, 0.05)

        for i in range(len(e.output_biases)):
            for j in range(len(e.output_biases[i])):
                if random.random() < 0.1:
                    e.output_biases[i][j] += random.uniform(-0.05, 0.05)
    
def train(elements, show_all_games, steps, gen_count, action_repeat = 4):
    cols = math.ceil(math.sqrt(len(elements)))
    for step in range(steps):
        obst_grids, frames = step_games_and_return_obstacles(elements, show_all_games)

        for e in elements:
            e.measure_fitness()

        if step % action_repeat == 0:
            for i, e in enumerate(elements): 
                e.obstacle_grid = obst_grids[i].flatten()
                e.forward(e.obstacle_grid, e.weights, e.biases)
                e.activation_ReLU(e.output)
                e.forward(e.output, e.hidden_layer_weights, e.output_biases)
                # print(e.output)
                e.generate_game_input()
        
        if show_all_games:
            leader_idx = max(range(len(elements)), key=lambda k: elements[k].fitness)
            render_with_feature(frames, cols, frames[leader_idx], fitness_history, gen_count)

        anyone_alive = False
        for e in elements:
            if e.game.alive_this_gen:
                anyone_alive = True
                break
        if not anyone_alive:
            print(f"all marios dead, gen {gen_count} over")
            break
    

    close_games(elements,show_all_games)

def draw_fitness_graph(history, w, h):
    graph = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.putText(graph, "best fitness:",(15,28),cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200),1,cv2.LINE_AA)
    if len(history) < 2:
        return graph
    low, high = min(history), max(history)
    span = (high - low) or 1
    pad = 40
    pts = []
    for i, f in enumerate(history):
        x = pad + int(i / (len(history) - 1) * (w - 2 * pad))
        y = (h - pad) - int((f - low) / span * (h - 2 * pad))
        pts.append((x, y))
    for a,b in zip(pts, pts[1:]):
        cv2.line(graph, a, b, (0,255,0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(graph,p,3,(0,255,0), -1)
    cv2.putText(graph,f"latest: {int(history[-1])}", (15,h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return graph

def render_with_feature(frames, cols, featured_frame, fitness_history, gen_count):
    grid = tile(frames, cols=cols)
    gh, gw, _ = grid.shape

    panel_h = gh // 2
    panel_w = int(panel_h * (256 / 240))

    featured = cv2.resize(featured_frame, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    graph = draw_fitness_graph(fitness_history, panel_w, panel_h)

    right_col = np.vstack([featured, graph])
    if right_col.shape[0] != gh:
        right_col = cv2.resize(right_col, (panel_w, gh))
    screen = np.hstack([grid, right_col])
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    cv2.putText(screen, f"GEN {gen_count}", (5, gh - 15), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0),20, cv2.LINE_AA)
    cv2.imshow('Mario', screen)
    cv2.waitKey(1)

def save_best(elements):
    best = max(elements, key=lambda e: e.fitness)
    np.savez("best_neural_network", weights=best.weights, biases = best.biases, hidden_layer_weights = best.hidden_layer_weights, output_biases = best.output_biases, fitness = best.fitness)
    print("saved best fitness")

def save_population(elements, gen_count, fitness_history):
    weights_list = []
    biases_list = []
    hidden_layer_weights_list = []
    output_biases_list = []
    fitness_list = []
    for e in elements:
        weights_list.append(e.weights)
        biases_list.append(e.biases)
        hidden_layer_weights_list.append(e.hidden_layer_weights)
        output_biases_list.append(e.output_biases)
        fitness_list.append(e.fitness)
    np.savez("last_population",
             weights=np.array(weights_list),
             biases=np.array(biases_list),
             hidden_layer_weights=np.array(hidden_layer_weights_list),
             output_biases=np.array(output_biases_list),
             fitness=np.array(fitness_list),
             gen_count = gen_count,
             fitness_history = np.array(fitness_history))
    print("saved last population")

def load_population(path="last_population.npz"):
    data = np.load(path)
    weights = data["weights"]
    biases = data["biases"]
    hidden_layer_weights = data["hidden_layer_weights"]
    output_biases = data["output_biases"]

    elements = []
    for i in range(len(weights)):
        e = Element(208, 12)
        e.weights = weights[i]
        e.biases = biases[i]
        e.hidden_layer_weights = hidden_layer_weights[i]
        e.output_biases = output_biases[i]
        e.fitness = 0
        elements.append(e)
    if "gen_count" in data:
        loaded_gen_count = int(data["gen_count"])
    else:
        loaded_gen_count = 0
    if "fitness_history" in data:
        loaded_fitness_history = list(data["fitness_history"])
    else:
        loaded_fitness_history = []
    return elements, loaded_gen_count, loaded_fitness_history

def load_top_from_population(path="last_population.npz"):
    data = np.load(path)
    fitness = data["fitness"]
    idx = int(np.argmax(fitness))

    e = Element(208,12)
    e.weights = data["weights"][idx]
    e.biases = data["biases"][idx]
    e.hidden_layer_weights = data["hidden_layer_weights"][idx]
    e.output_biases = data["output_biases"][idx]
    e.fitness = 0
    return e

steps = 2000
n = 20
best_fitness = []
rolling_avg_fitness = []

if os.path.exists("last_population.npz"):
    elements, gen_count, fitness_history = load_population()
    print("Last pop was loaded")
else:
    elements = create_population(n)
    gen_count = 0
    fitness_history = []
    print("No population found, starting new")
start_time = time.time()
duration = 24 * 60 * 60

try:
    while time.time() - start_time < duration:
        train(elements=elements,show_all_games=True,steps=steps, gen_count=gen_count)
        best = max(e.fitness for e in elements)
        fitness_history.append(best)
        print(f"gen: {gen_count} | best fitness: {best}")
        mating_pool = generate_mating_pool()
        elements, n_of_elites = breed_new_pop(n, mating_pool, elements)
        mutate_population(elements, n_of_elites)
        gen_count += 1
        save_best(elements)
        save_population(elements,gen_count, fitness_history)

finally:
    save_best(elements)
    save_population(elements,gen_count, fitness_history)