#!/usr/bin/python3

from TSPClasses import *
from State import *
from MyQueue import *
from Genome import *
import time
import numpy as np
import multiprocessing as mp
from copy import deepcopy


class TSPSolver:
	def __init__(self):
		self._scenario = None
		self.generation_fitness = 0
		self.generation = 0
		self.cap = 100000

	def setup_with_scenario(self, scenario):
		self._scenario = scenario

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def default_random_tour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		num_cities = len(cities)
		found_tour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not found_tour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(num_cities)
			route = []
			# Now build the route using the random permutation
			for i in range(num_cities):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				found_tour = True
		end_time = time.time()
		results['cost'] = bssf.cost if found_tour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	# This function is O(n^3) because it contains three nested loops each of which run in O(n) time
	def greedy(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		num_cities = len(cities)
		found_tour = False
		count = 0
		bssf = None
		start_time = time.time()

		# pick the first starting city
		starting_city_index = 0

		# keep looking for solutions until we find one or run out of time
		# worst case scenario, all cities are used as the starting city, and the loop is O(n)
		while not found_tour and time.time() - start_time < time_allowance and starting_city_index < num_cities:

			# put the starting city on the partial path
			route = [cities[starting_city_index]]

			# create a list of cities not yet on the path (i.e., all city but the starting one)
			remaining_cities = deepcopy(cities)		# O(n)
			remaining_cities.pop(starting_city_index)	 # O(n)

			# add a new city to the partial path until all the cities are on the path
			path_is_valid = True
			# worst case scenario, all the cities are added to the path, and the loop is O(n)
			while len(remaining_cities) > 0 and path_is_valid:

				# the current city is the one most recently added to the partial path
				current_city = route[-1]

				# create variables to keep track of the closest city and how far away it is
				closest_city = None
				closest_distance = np.inf

				# worst case scenario, all the cities remain and must be checked, making the loop O(n)
				for city in remaining_cities:
					if current_city.cost_to(city) < closest_distance:
						closest_city = city
						closest_distance = current_city.cost_to(city)

				# if there were no edges leaving the current city, start over
				if closest_city is None:
					path_is_valid = False
					continue

				# if there is an edge leaving the current city,
				# add the closest neighbor to path and remove it from the list of cities
				route.append(closest_city)	 # O(1)
				remaining_cities.remove(closest_city)	 # worst case, O(n)

			# O(n)
			solution = TSPSolution(route)

			# if the loop ended because there wasn't a valid path, increment the starting city and start over
			if not path_is_valid or len(route) != num_cities or solution.cost >= np.inf:
				starting_city_index += 1
				continue

			# if the path is valid save it as the bssf, and increment the count
			bssf = solution
			found_tour = True
			count += 1

		end_time = time.time()
		results['cost'] = bssf.cost if found_tour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	# This function is O(n!n^3) because
	# a O(n!) loop visiting every possible state contains
	# a O(n) loop checking every city, which contains
	# a O(n^2) function to deep copy the entire matrix
	def branch_and_bound(self, time_allowance=60.0):

		start_time = time.time()

		city_list = self._scenario.getCities()

		# calculate an initial solution O(n^3)
		results = self.greedy()

		# set up fields for results
		best_solution = results['soln']
		best_cost = results['cost']
		num_solutions = 0
		num_states = 0
		num_pruned = 0

		# build the initial state
		initial_cost_matrix = self.build_matrix		# O(n^2)
		city_indices_list = list(range(len(city_list)))		# O(n)
		initial_state = State(initial_cost_matrix, 0, list(), city_indices_list)
		initial_state.reduce_matrix()		# O(n)

		# set up the queue
		my_queue = MyQueue()
		my_queue.insert(initial_state)	 # O(nlog(n))

		# search for solutions until we run out of time or try every option
		# worst case scenario, every possible state is added to queue, making the loop O(n!)
		while time.time() - start_time < time_allowance and not my_queue.is_empty():

			# pull the best state off of the queue O(nlog(n))
			current_state = my_queue.delete_min()

			# if the lower bound is greater than the best cost so far, skip this state
			if current_state.lower_bound > best_cost:
				num_pruned += 1
				continue

			# create a child for each remaining unused city
			# worst case scenario, all cities remain, and the loop is O(n)
			for i in range(len(current_state.unused_city_indices)):

				# make all the parameters for the child state
				child_matrix = deepcopy(current_state.matrix)	 # O(n^2)
				child_unused_city_indices = deepcopy(current_state.unused_city_indices)		# O(n)
				next_city_index = child_unused_city_indices.pop(i)	 # worst case, O(n)
				child_partial_path = deepcopy(current_state.partial_path)	 # worst case, O(n)
				child_partial_path.append(next_city_index)	 # O(1)

				# create the child state
				child_state = State(child_matrix, current_state.lower_bound, child_partial_path, child_unused_city_indices)
				num_states += 1

				# if the child state's partial path contains any edges, deal with the most recent one
				if child_state.get_depth() >= 2:

					# identify the most recently added edge
					from_index = child_state.partial_path[-2]
					to_index = child_state.partial_path[-1]

					# reduce the cost matrix in accordance with the most recently added edge
					child_state.infinitize(from_index, to_index)	 # O(n)
					child_state.reduce_matrix()		# O(n^2)

				# if the child's lower bound is less than the best cost so far, add it to the queue
				if child_state.lower_bound < best_cost:
					my_queue.insert(child_state)	 # O(nlog(n))
				else:
					num_pruned += 1

			# if all the cities have been added to the path
			if len(current_state.unused_city_indices) == 0:

				# identify the final edge and add its cost
				from_index = current_state.partial_path[-1]
				to_index = current_state.partial_path[0]
				current_state.infinitize(from_index, to_index)	  # O(n)

				# if the result is better than our best so far, save it
				if current_state.lower_bound < best_cost:
					best_cost = current_state.lower_bound
					route = [city_list[index] for index in current_state.partial_path]		# O(n))
					best_solution = TSPSolution(route)		# O(n)
					num_solutions += 1

		end_time = time.time()

		# prune all states remaining on the queue
		num_pruned += my_queue.get_size()

		results['cost'] = best_solution.cost
		results['time'] = end_time - start_time
		results['count'] = num_solutions
		results['soln'] = best_solution
		results['max'] = my_queue.get_max_size()
		results['total'] = num_states
		results['pruned'] = num_pruned

		return results

	@property
	# O(n^2)
	def build_matrix(self):

		cities = self._scenario.getCities()
		num_cities = len(cities)

		# O(n^2) to build matrix
		cost_matrix = [[np.inf] * num_cities for i in range(num_cities)]

		# O(n^2) to fill matrix
		for i in range(num_cities):
			for j in range(num_cities):
				if i != j:
					cost_matrix[i][j] = cities[i].cost_to(cities[j])

		return cost_matrix

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def fancy(self, time_allowance=60.0):
		start_time = time.time()
		# population size
		k = 100
		num_children = k - 2
		num_keep = k - num_children

		num_genes = len(self._scenario.getCities())

		# generate initial population
		initial_population = self.generate_initial_population(k, method_name='greedy')

		while self.generation < time_allowance:
			self.generation += 1
			# calculate population fitness
			self.calculate_population_fitness(initial_population)

			#print('\nInitial Population:')
			#for genome in initial_population:
				#print(genome.path, genome.fitness)
				#assert self.get_fitness(genome) != np.inf
			#print('Initial Population Size:', len(initial_population))

			# TODO select parents
			parents = self.select_parents(initial_population, num_children)

			# TODO crossover
			crossover_children = self.crossover(parents, num_genes//2)

			# mutate population
			# TODO only mutate children
			self.mutate_population(crossover_children, chance_of_mutating=7, num_mutations=1)

			#print('\nMutated Population')
			#for genome in  crossover_children:
				#print(genome.path)
				#assert self.get_fitness(genome) != np.inf
			#print('Mutated Population Size:', len(crossover_children))

			# cull population
			# TODO only cull parents
			initial_population = self.cull_population(initial_population, method_name='random', num_to_keep=num_keep, top_to_keep= 1)
			crossover_children.extend(initial_population)
			initial_population = crossover_children
			#print('\nCulled Population')
			#for genome in initial_population:
				#print(genome.path, genome.fitness)
			#	assert self.get_fitness(genome) != np.inf
			#print('Culled Population Size:', len(initial_population))

	# This function is O(n^3) because it contains three nested loops each of which run in O(n) time
	def all_greedy_paths(self, max_tours=None):

		# set up initial fields
		results = []
		cities = self._scenario.getCities()
		num_cities = len(cities)
		tours_found = 0
		if max_tours is None:
			max_tours = num_cities

		# pick the first starting city
		starting_city_index = 0

		# keep looking for solutions until we find one or run out of time
		# worst case scenario, all cities are used as the starting city, and the loop is O(n)
		while tours_found < max_tours and starting_city_index < num_cities:

			# put the starting city on the partial path
			route_indices = [starting_city_index]

			# create a list of cities not yet on the path (i.e., all city but the starting one)
			remaining_cities_indices = list(range(len(cities)))
			remaining_cities_indices.pop(starting_city_index)

			# add a new city to the partial path until all the cities are on the path
			path_is_valid = True
			# worst case scenario, all the cities are added to the path, and the loop is O(n)
			while len(remaining_cities_indices) > 0 and path_is_valid:

				# the current city is the one most recently added to the partial path
				current_city_index = route_indices[-1]

				# create variables to keep track of the closest city and how far away it is
				closest_city_index = None
				closest_distance = np.inf

				# worst case scenario, all the cities remain and must be checked, making the loop O(n)
				for city_index in remaining_cities_indices:
					if cities[current_city_index].cost_to(cities[city_index]) < closest_distance:
						closest_city_index = city_index
						closest_distance = cities[current_city_index].cost_to(cities[city_index])

				# if there were no edges leaving the current city, start over
				if closest_city_index is None:
					path_is_valid = False
					continue

				# if there is an edge leaving the current city,
				# add the closest neighbor to path and remove it from the list of cities
				route_indices.append(closest_city_index)	 # O(1)
				remaining_cities_indices.remove(closest_city_index)	 # worst case, O(n)

			# O(n)
			route = [cities[index] for index in route_indices]
			solution = TSPSolution(route)

			# if the loop ended because there wasn't a valid path, increment the starting city and start over
			if not path_is_valid or len(route) != num_cities or solution.cost >= np.inf:
				starting_city_index += 1
				continue

			# if the path is valid save it as the bssf, and increment the count
			genome = Genome(route_indices)
			results.append(genome)
			tours_found += 1
			starting_city_index += 1

		return results

	def greedy_initial_population(self, population_size):

		num_cities = len(self._scenario.getCities())
		cutoff = min(num_cities, population_size)

		greedy_population = self.all_greedy_paths(cutoff)

		if len(greedy_population) < population_size:
			difference = population_size - len(greedy_population)
			greedy_population += self.random_initial_population(difference)

		return greedy_population

	def generate_initial_population(self, population_size, method_name):
		if method_name == 'greedy':
			return self.greedy_initial_population(population_size)
		elif method_name == 'random':
			return self.random_initial_population(population_size)
		else:
			print('unrecognized method for generating initial population')
			assert False

	def random_path(self):
		cities = self._scenario.getCities()
		num_cities = len(cities)
		while True:
			perm = np.random.permutation(num_cities)
			route = [cities[i] for i in perm]
			solution = TSPSolution(route)
			if solution.cost < np.inf:
				return list(perm)

	def random_initial_population(self, population_size):
		random_population = list()
		for i in range(population_size):
			random_population.append(Genome(self.random_path()))
		return random_population

	# def random_initial_population(self, population_size):
	# 	self.parallel_pool = mp.Pool(mp.cpu_count())
	# 	random_population = list()
	# 	for i in range(population_size):
	# 		self.parallel_search_result = None
	# 		[self.parallel_pool.apply_async(self.random_path, args=None, callback=self.parallel_search_callback) for i in range(10)]
	# 		self.parallel_pool.close()
	# 		while self.parallel_search_result is None:
	# 			time.sleep(2)
	# 		random_population.append(Genome(self.parallel_search_result))
	# 	return random_population
	#
	# def parallel_search_callback(self, result):
	# 	if result:
	# 		self.parallel_search_result = result
	# 		self.pool.terminate()

	def get_fitness(self, genome):
		solution = TSPSolution([self._scenario.getCities()[i] for i in genome.path])
		if solution.cost == np.inf:
			return 0
		return 1/solution.cost

	def calculate_population_fitness(self, population):
		self.generation_fitness = 0
		for genome in population:
			genome.fitness = self.get_fitness(genome)
			self.generation_fitness += genome.fitness

	def select_parents(self, population, num_parents=2):
		winners = []
		for i in range(num_parents):
			winners.append(self.roulette_select(population))
		return winners

	def tournament_select(self, population, tournament_size=3):
		tourn = []
		for i in range(tournament_size):
			chromosome = None
			while chromosome is None or chromosome in tourn:
				chromosome = population[random.randint(0, len(population)-1)]
			tourn.append(chromosome)
		winner = tourn[0]
		for chromosome in tourn:
			if chromosome.fitness > winner.fitness:
				winner = chromosome
		return winner

	def roulette_select(self, population):
		threshold = random.uniform(0, self.generation_fitness)
		partial_sum = population[0].fitness
		index = 0
		while partial_sum < threshold:
			index += 1
			partial_sum += population[index].fitness
		return population[index]

	def crossover_parents(self, parent1, parent2, num_genes=1):
		total_cities = len(self._scenario.getCities())

		# crossover slice will be num_genes long
		non_crossover = total_cities - num_genes
		start = random.randint(0, non_crossover)
		end = start + num_genes

		parent1_slice = parent1.path[start:end]
		parent2_slice = parent2.path[start:end]

		child1 = []
		child2 = []
		count = 0
		for i in parent2.path:
			if count == non_crossover:
				break
			if i not in parent1_slice:
				count += 1
				child1.append(i)

		count = 0
		for i in parent1.path:
			if count == non_crossover:
				break
			if i not in parent2_slice:
				count += 1
				child2.append(i)
		child1[start:start] = parent1_slice
		child2[start:start] = parent2_slice
		return child1, child2

	def crossover(self, parents, num_genes=1):
		children = []
		for i in range(0, len(parents), 2):
			c1, c2 = self.crossover_parents(parents[i], parents[i+1], num_genes)
			children.append(Genome(c1))
			children.append(Genome(c2))

		return children

	def mutate_genome(self, genome, num_mutations=1, return_valid_path=False):
		while True:
			# path = deepcopy(genome.path)

			# mutate the correct number of times
			for i in range(num_mutations):
				# get two indices so we can swap them
				while True:
					index_1 = random.randint(0, len(genome.path) - 1)
					index_2 = random.randint(0, len(genome.path) - 1)
					if index_1 != index_2:
						break
				genome.path[index_1], genome.path[index_2] = genome.path[index_2], genome.path[index_1]
				# swap the two values
				# temp = path[index_1]
				# path[index_1] = path[index_2]
				# path[index_2] = temp

			# if it doesn't matter if the path is valid, break
			if not return_valid_path:
				break

			# test if the path is valid; if so, break
			#route = [self._scenario.getCities()[i] for i in path]
			#solution = TSPSolution(route)
			#if solution.cost < np.inf:
			#	break

	def mutate_genome_test(self, genome, index_1, num_mutations=1, return_valid_path=False):
		while True:
			# mutate the correct number of times
			for i in range(num_mutations):
				# get two indices so we can swap them
				while True:
					#index_1 = random.randint(0, len(genome.path) - 1)
					index_2 = random.randint(0, len(genome.path) - 1)
					if index_1 != index_2:
						break
				genome.path[index_1], genome.path[index_2] = genome.path[index_2], genome.path[index_1]

			# if it doesn't matter if the path is valid, break
			if not return_valid_path:
				break

	def mutate_population(self, population, chance_of_mutating=25, num_mutations=1, return_valid_path=False):
		for genome in population:
			for i, gene in enumerate(genome.path):
				if random.randint(0, 99) < chance_of_mutating:
					self.mutate_genome_test(genome, i, num_mutations, return_valid_path)
			#if random.randint(0, 99) < chance_of_mutating:
			#	self.mutate_genome(genome, num_mutations, return_valid_path)

	def random_cull(self, population, num_to_keep, top_to_keep=1):

		old_population = deepcopy(population)
		new_population = list()

		for i in range(top_to_keep):
			top_index = self.get_index_of_best_genome(old_population)
			new_population.append(old_population[top_index])
			old_population.pop(top_index)

		while len(new_population) < num_to_keep:
			random_index = random.randint(0, len(old_population) - 1)
			new_population.append(old_population[random_index])
			old_population.pop(random_index)

		return new_population

	@staticmethod
	def ranked_cull(population, num_to_keep):
		return sorted(population, key=lambda genome: genome.fitness, reverse=False)[:num_to_keep]

	def cull_population(self, population, num_to_keep, method_name, top_to_keep=1):
		if method_name == 'ranked':
			culled = self.ranked_cull(population, num_to_keep)
			print("Gen " + str(self.generation) + " Champion - Fitness: " + str(culled[0].fitness) + " - Cost: " + str(1/culled[0].fitness))
			return culled
		elif method_name == 'random':
			culled = self.random_cull(population, num_to_keep, top_to_keep)
			print("Gen " + str(self.generation) + " Champion - Fitness: " + str(culled[0].fitness) + " - Cost: " + str(1/culled[0].fitness))
			return culled
		# TODO implement roulette culling method
		else:
			print('unrecognized method for culling population')
			assert False

	@staticmethod
	def get_index_of_best_genome(population):
		best_fitness = float('-inf')
		best_index = None

		for i in range(len(population)):
			if population[i].fitness > best_fitness:
				best_fitness = population[i].fitness
				best_index = i

		return best_index
