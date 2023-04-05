import sys
import numpy as np
import random
import time

class Problem:
    def random_solution(self) -> list:
        return []

    def evaluate(self, solution) -> float:
        return 0.0

class HASHCODE_PROBLEM(Problem):
    def __init__(self, days, number_of_libraries, libraries, books):
        self.days = days
        self.number_of_libraries = number_of_libraries
        self.libraries = libraries
        self.books = books

    def random_solution(self):
        solution = []
        days_left = self.days
        for library in np.random.permutation(self.libraries):
            new_library = Library(library.id, library.number_of_books, library.books, library.signup_time, library.books_per_day)
            days_left = max(days_left - library.signup_time, 0)
            new_library.days_left = days_left
            new_books = new_library.choose_random_books()
            solution.append((new_library, new_books))
        return solution

    def evaluate(self, solution) -> int:
        set_of_books = set()
        for library, books in solution:
            set_of_books.update(books)
        return sum(book.score for book in set_of_books)

class SolutionEvolutionary():
    def __init__(self, problem, size_of_population = 10, mutation_probability = 0.4, crossover_probability = 0.2, tournament_size = 0.2):
        self.problem = problem
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.tournament_size = tournament_size
        self.population_size = size_of_population
        self.population = self.initialize_greedy_population()

    def initialize_population(self):
        population = []
        while len(population) < self.population_size:
            solution = self.problem.random_solution()
            population.append((solution, self.problem.evaluate(solution)))
        return population
    
    def omega(self, library):
        return library.books_per_day*sum([book.score for book in library.books])/library.number_of_books
    
    def fi(self, library):
        return sum([book.score for book in library.books])/library.signup_time

    def gamma(self, library):
        return 1/library.signup_time

    def initialize_greedy_population(self):
        population = []
        NEWLIBRARIES = sorted(self.problem.libraries, key = lambda x: (self.fi(x)), reverse = True)
        while len(population) < self.population_size:
            solution = []
            read = set()
            days_left = self.problem.days
            for library in NEWLIBRARIES:
                new_library = Library(library.id, library.number_of_books, library.books, library.signup_time, library.books_per_day)
                days_left = max(days_left - library.signup_time, 0)
                new_library.days_left = days_left
                new_books = new_library.choose_greedy_books(read)
                solution.append((new_library, new_books))
            population.append((solution, self.problem.evaluate(solution)))
        return population

    def mutate(self, solution):
        if np.random.uniform(0, 1) > self.mutation_probability:
            return solution
        
        for i in range(self.problem.number_of_libraries // 2):
            if np.random.uniform(0, 1) <= 10 * self.mutation_probability / self.problem.number_of_libraries:
                index1, index2 = random.sample(range(len(solution)), 2)
                solution[index1], solution[index2] = solution[index2], solution[index1]

        days_checker = self.problem.days
        new_solution = []
        for library, books in solution:
            days_checker = max(days_checker - library.signup_time, 0)
            library.days_left = days_checker
            new_books = library.fix_books(books)

            if np.random.uniform(0, 1) <= self.mutation_probability:
                new_books = library.mutate_books(new_books)
            new_solution.append((library, new_books))
        
        return new_solution
    
    def crossover_books(self, s1, s2, dictionary):

        new_solution = []
        days_checker = self.problem.days
            
        for library1, books1 in s1:
            if np.random.uniform(0, 1) > self.crossover_probability:
                days_checker = max(days_checker - library1.signup_time, 0)
                library1.days_left = days_checker
                new_books = library1.fix_books(books1)
                new_solution.append((library1, new_books))
                continue
            
            library2, books2 = s2[dictionary[library1.id]]
            intersection = books1.intersection(books2)

            n1 = len(books1) - len(intersection)
            n2 = len(books2) - len(intersection)

            if n2 >= n1//2:
                intersection.update(random.sample((books1-intersection), n1//2) + random.sample((books2-intersection), n1//2))

                days_checker = max(days_checker - library1.signup_time, 0)
                library1.days_left = days_checker
                new_books = library1.fix_books(intersection)

                new_solution.append((library1, new_books))    
            else:
                difference = n1 - n2
                intersection.update(random.sample((books1-intersection), difference) + random.sample((books2-intersection), n2))

                days_checker = max(days_checker - library1.signup_time, 0)
                library1.days_left = days_checker
                new_books = library1.fix_books(intersection)

                new_solution.append((library1, new_books))
        return new_solution

    def crossover(self, s1, s2):
        if np.random.uniform(0, 1) > self.crossover_probability:
            return s1, s2

        solution1 = []
        solution2 = []
        solution1_ids = set()
        dict1 = dict()
        dict2 = dict()

        for (l1, b1), (l2, b2) in zip(s1, s2):
            if l1.id not in solution1_ids:
                solution1.append((l1, b1))
                solution1_ids.add(l1.id)
                dict1[l1.id] = len(solution1) - 1
            else:
                solution2.append((l1, b1))
                dict2[l1.id] = len(solution2) - 1

            if l2.id not in solution1_ids:
                solution1.append((l2, b2))
                solution1_ids.add(l2.id)
                dict1[l2.id] = len(solution1) - 1
            else:
                solution2.append((l2, b2))
                dict2[l2.id] = len(solution2) - 1
        
        new_solution1 = self.crossover_books(solution1, solution2, dict2)
        new_solution2 = self.crossover_books(solution2, solution1, dict1)
        return new_solution1, new_solution2
    
    def copy_solution(self, solution):
        new_solution = []
        for library, books in solution:
            new_library = Library(library.id, library.number_of_books, library.books, library.signup_time, library.books_per_day)
            new_library.days_left = library.days_left
            new_books = set(books)
            new_solution.append((new_library, new_books))
        return new_solution

    def tournament_round(self, population):
        tournament = random.sample(population, max(1, int(self.tournament_size * self.population_size)))
        return self.copy_solution(tournament[0][0])

    def evolution_step(self):
        winners = []
        for i in range(int(self.tournament_size * self.population_size)):
            solution = (self.copy_solution(self.population[i][0]), self.population[i][1])
            winners.append(solution)

        while len(winners) < self.population_size:
            winner1, winner2 = self.tournament_round(self.population), self.tournament_round(self.population)
            winner1, winner2 = self.crossover(winner1, winner2)
            winner1 = self.mutate(winner1)
            winner2 = self.mutate(winner2)
            winners.append((winner1, self.problem.evaluate(winner1)))
            winners.append((winner2, self.problem.evaluate(winner2)))
        
        return sorted(winners, key=lambda x: x[1], reverse = True)

    def evolution(self, maximum, start_time, best_solution = None, execution_time = 0, no_improvement = 0):
        for epoch, _ in enumerate(iter(bool, True)):
            if execution_time > 240 or no_improvement > 480:
                return maximum, best_solution, execution_time
            
            self.population = self.evolution_step()

            if self.population[0][1] > maximum:
                no_improvement = 0
                best_solution = self.population[0][0]
                maximum = self.population[0][1]
            
            no_improvement += 1
            execution_time = time.time() - start_time


class Library:
    days_left = None

    def __init__(self, id, number_of_books, books, signup_time, books_per_day):
        self.id = id
        self.number_of_books = number_of_books
        self.books = books
        self.signup_time = signup_time
        self.books_per_day = books_per_day

    def choose_random_books(self):
        if self.days_left == 0:
            return set()
        books = set(random.sample(self.books, min(self.number_of_books, self.books_per_day * self.days_left)))
        return books

    def choose_greedy_books(self, read):
        chosen = set()
        if self.days_left == 0:
            return set()

        books = sorted(list(self.books), key = lambda x: x.score, reverse = True)
        for book in books:
            if len(chosen) == self.days_left*self.books_per_day:
                return chosen
                
            if book not in read:
                chosen.add(book)
                read.add(book)
        return chosen

    def mutate_books(self, chosen_books):
        if len(chosen_books) == self.number_of_books or len(chosen_books) == 0:
            return chosen_books
        
        books_to_choose_from = self.books - chosen_books
        n = np.random.randint(1, min(len(books_to_choose_from), len(chosen_books))+1)

        books_to_remove = random.sample(chosen_books, n)
        books_to_add = random.sample(books_to_choose_from, n)

        chosen_books.difference_update(books_to_remove)
        chosen_books.update(books_to_add)        
        return chosen_books

    def fix_books(self, chosen_books):
        if self.days_left == 0:
            return set()
        
        delta = min(self.days_left * self.books_per_day, self.number_of_books) - len(chosen_books)

        if delta > 0:
            chosen_books.update(random.sample(self.books - chosen_books, delta))
            return chosen_books
        
        if delta < 0:
            chosen_books.difference_update(random.sample(chosen_books, -delta))
            return chosen_books
        
        return chosen_books

class Book:
    def __init__(self, id, score):
        self.id = id
        self.score = score

def write_to_file(solution):
    sys.stdout.write(str(NUMBER_OF_LIBRARIES))
    sys.stdout.write('\n')
    for library, books in solution:
        sys.stdout.write(str(library.id))
        sys.stdout.write(" ")
        sys.stdout.write(str(len(books)))
        sys.stdout.write('\n')
        books = " ".join(str(book.id) for book in books)
        sys.stdout.write(books)
        sys.stdout.write('\n')

if __name__ == '__main__':
    start_time = time.time()
    lines = sys.stdin.read().splitlines()

    NUMBER_OF_BOOKS, NUMBER_OF_LIBRARIES, NUMBER_OF_DAYS = (int(x) for x in lines.pop(0).split())
    BOOKS = tuple(Book(int(id), int(score)) for id, score in enumerate(lines.pop(0).split()))

    LIBRARIES = []
    for library in range(NUMBER_OF_LIBRARIES):
        number_of_books, signup_time, books_per_day = (int(x) for x in lines.pop(0).split())
        books = set(BOOKS[int(id)] for id in lines.pop(0).split())
        LIBRARIES.append(Library(library, number_of_books, books, signup_time, books_per_day))

    HASH_CODE = HASHCODE_PROBLEM(NUMBER_OF_DAYS, NUMBER_OF_LIBRARIES, LIBRARIES, books)
    SOLUTION = SolutionEvolutionary(HASH_CODE, 16, 0.6, 0.2, 0.25)
    EVAL, best_solution, execution_time = SOLUTION.evolution(0, start_time)
    write_to_file(best_solution)