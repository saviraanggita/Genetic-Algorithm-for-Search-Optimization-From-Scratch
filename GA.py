import numpy as np
import random as rand
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Fungsi untuk melakukan binary encoding dengan mengubah bilangan biner menjadi bilangan real
# Fungsi mengembalikan 2 buah nilai x1 dan x2 berbentuk bilangan real dari gene1 dan gene2 yang berbentuk bilangan biner
def binary_encoding(chromosome):
    gene1 = chromosome[0]
    gene2 = chromosome[1]
    b,g1,g2 = 0,0,0

    for i in range (len(gene1)):
        b += 2**(-i)
        g1 += int(gene1[i])*2**(-i)
        g2 += int(gene2[i])*2**(-i)

    x1 = (-3) + (((3-(-3))/b)*(g1))
    x2 = (-2) + (((2-(-2))/b)*(g2))

    return x1,x2

# Fungsi untuk membuat kromosom yang terdiri dari 2 gen (masing-masing gen menampung nilai x1 dan x2)
# Fungsi mengembalikan sebuah list kromosom yang terdiri dari 2 gen yang menampung nilai x1 dan x2
def create_chromosome(bit_chromosome):
    chromosome = []
    gene1,gene2 = "",""

    # Nilai bit kromosom harus genap agar tiap2 gen mendapat jumlah bit yang sama
    if (bit_chromosome%2==0):
        for i in range (bit_chromosome//2):
            gene1 += str(rand.randint(0,1))
            gene2 += str(rand.randint(0,1))
        chromosome = [gene1,gene2]
    
    return chromosome

# Fungsi untuk membuat populasi yang terdiri dari kromosom (individu) dimana jumlah bit kromosom disesuaikan dengan nilai input
# Fungsi mengembalikan sebuah list populasi yang didalamnya terdapat list kromosom individu (list 2 dimensi)
def create_population(num_population,bit_chromosome):
    population = []
    for i in range (num_population):
        individual = create_chromosome(bit_chromosome)
        population.append(individual)
    return population

# Fungsi yang akan digunakan dalam penerapan metode pencarian nilai minimum fungsi dengan algoritma genetika
def function(x1,x2):
    funct = ((4-2.1*(x1**2)+((x1**4)/3))*x1**2) + (x1*x2) + ((-4+4*(x2**2))*x2**2)
    return funct

# Fungsi untuk mengevaluasi nilai kualitas (fitness) individu dalam sebuah populasi
# Fungsi mengembalikan list nilai fitness dari tiap2 individu dalam populasi
def fitness_evaluation(population):
    L_fitness = []
    a = 3
    for i in range (len(population)):
        x1,x2 = binary_encoding(population[i])
        h = function(x1,x2)
        fitness = 1/(h+a)
        L_fitness.append(fitness)
    return L_fitness

# Fungsi untuk melakukan seleksi individu terbaik dalam sebuah populasi dengan metode turnamen
# Fungsi mengembalikan individu yang menjadi pemenang turnamen untuk dijadikan sebagai parent
def tournament(population,L_fitness):
    # Memilih 2 individu secara random untuk turnamen
    individual1 = rand.randint(0, len(L_fitness)-1)
    individual2 = rand.randint(0, len(L_fitness)-1)
    
    # Menampung nilai fitness dari individu 1 dan 2
    fitness1 = L_fitness[individual1]
    fitness2 = L_fitness[individual2]
    
    # Membandingkan nilai fitness antara individu 1 dan 2 dimana nilai tertinggi menjadi pemenang turnamen
    if (fitness1>=fitness2):
        winner = individual1
    else:
        winner = individual2
    
    # Mengembalikan individu pemenang yang nantinya akan menjadi parent
    return population[winner]

# Fungsi untuk melakukan persilangan kromosom antara parent1 dan parent2 dengan probabilitas tertentu
# Fungsi mengembalikan 2 offspring/child sebagai hasil persilangan dari kedua parent 
def crossover(parent1,parent2,crossover_probability):
    # Mengambil nilai secara random untuk dibandingkan dengan probabilitas crossover (Pc)
    random_values = rand.uniform(0.0,1.0)
    if (random_values<=crossover_probability):
        chromosome_length = len(parent1[0])+len(parent1[1])
        
        # Memilih titik crossover secara random
        crossover_point = rand.randint(1,chromosome_length-1)

        # Menggabungkan gen1 dan gen2 dari tiap parent menjadi sebuah string agar mempermudah proses persilangan kromosom
        string_p1 = parent1[0]+parent1[1]
        string_p2 = parent2[0]+parent2[1]

        # Menyilangkan kromosom parent1 dan parent2 berdasarkan titik crossover tertentu dan menghasilkan string kromosom child
        string_c1 = string_p1[:crossover_point]+string_p2[crossover_point:]
        string_c2 = string_p2[:crossover_point]+string_p1[crossover_point:]

        # string kromosom child diubah bentuknya menjadi list berisi gen1 dan gen2 dari child
        child1 = [string_c1[:chromosome_length//2],string_c1[chromosome_length//2:]]
        child2 = [string_c2[:chromosome_length//2],string_c2[chromosome_length//2:]]

    else:
        child1 = parent1
        child2 = parent2
    
    return child1, child2

# Fungsi untuk melakukan mutasi dari sebuah individu dengan probabilitas tertentu
# Fungsi mengembalikan individu hasil mutasi
def mutation(individual, mutation_probability):
    # Mengambil nilai secara random untuk dibandingkan dengan probabilitas mutasi (Pm)
    random_values = rand.uniform(0.0,1.0)
    if (random_values<=mutation_probability):
        string = individual[0]+individual[1]

        # Mengambil nilai secara random untuk menentukan posisi mana yang akan dimutasi pada kromosom
        pos = rand.randint(0,len(string)-1)
        if (int(string[pos])==1):
            string = string[:pos]+'0'+string[pos+1:] # Mutasi dengan mengganti nilai 1 menjadi 0
        else:
            string = string[:pos]+'1'+string[pos+1:] # Mutasi dengan mengganti nilai 0 menjadi 1

        individual = [string[:len(string)//2],string[len(string)//2:]]
    else:
        individual = individual

    return individual

# Proses Inisialisasi
L_best_fitness = []             # List untuk menampung nilai fitness terbaik
L_best_result = []              # List untuk menampung solusi (nilai minimum) terbaik
L_best_individual = []          # List untuk menampung individu2 dengan nilai fitness terbaik
L_x1 = []                       # List untuk menampung nilai x1 dari individu2 dengan nilai fitness terbaik
L_x2 = []                       # List untuk menampung nilai x2 dari individu2 dengan nilai fitness terbaik
new_population = []             # List untuk menampung populasi baru dalam perulangan

bit_chromosome = 20              # nilai jumlah bit kromosom yang digunakan
num_generation = 20             # nilai jumlah generasi untuk membangkitkan populasi
num_population = 500            # nilai jumlah populasi yang akan dibangkitkan
mutation_probability = 0.01     # nilai probabilitas mutasi (Pm)
crossover_probability = 0.65    # nilai probabilitas crossover (Pc)

# Pembuatan populasi pertama
population = create_population(num_population,bit_chromosome)
fitness = fitness_evaluation(population)

L_best_fitness.append(np.max(fitness))

index = fitness.index(np.max(fitness))
L_best_individual.append(population[index])

x1,x2 = binary_encoding(population[index])
best_result = function(x1,x2)
L_best_result.append(best_result)
L_x1.append(x1)
L_x2.append(x2)

#Memulai perulangan algoritma genetik dengan seleksi survivor Generational Replacement
for g in range (num_generation):
    new_population = []

    #Populasi meregenerasi 2 child dalam satu waktu
    for p in range(int(num_population/2)):
        # Melakukan seleksi individu dengan metode turnamen untuk menghasilkan parent
        parent1 = tournament(population, fitness)
        parent2 = tournament(population, fitness)

        # Melakukan persilangan kromosom dengan probabilitas tertentu antara 2 parent untuk menghasilkan child
        child1,child2 = crossover(parent1, parent2, crossover_probability)

        # Mutasi child dengan probabilitas tertentu
        child1 = mutation(child1,mutation_probability)
        child2 = mutation(child2,mutation_probability)
        
        new_population.append(child1)
        new_population.append(child2)
    
    population = new_population
    fitness = fitness_evaluation(population)
    
    # Menyimpan nilai fitness terbaik dari populasi pada generasi g ke dalam list fitness terbaik
    L_best_fitness.append(np.max(fitness))

    # Menyimpan individu yang memiliki fitness terbaik dalam populasi pada generasi g ke dalam list individu terbaik
    index = fitness.index(np.max(fitness))
    L_best_individual.append(population[index])

    # Menyimpan nilai minimum yang dihasilkan oleh individu terbaik ke dalam list hasil nilai minimum terbaik
    x1,x2 = binary_encoding(population[index])
    best_result = function(x1,x2)
    L_best_result.append(best_result)
    L_x1.append(x1)
    L_x2.append(x2)

# Menampilkan solusi optimum saat running program    
print('Best Fitness\t\t\t: ',np.max(L_best_fitness))
print('Best Minimum Value\t\t: ',np.min(L_best_result))

index = L_best_fitness.index(np.max(L_best_fitness))
print('Chromosome (genotype)\t: ','gene 1 (x1) = ',L_best_individual[index][0],', gene 2 (x2) = ',L_best_individual[index][1])

x1,x2 = binary_encoding(L_best_individual[index])
print('Chromosome (phenotype)\t: gene 1 (x1) = ',x1,', gene 2 (x2) = ',x2)

# Plot fungsi dan nilai minimum dalam bentuk grafik kontur 3 dimensi
plt.figure()
ax = plt.axes(projection='3d')
x_ax = np.linspace(-6, 6, 30)
y_ax = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x_ax, y_ax)
Z = function(X, Y)
z = function(x1,x2)
ax.contour3D(X, Y, Z, 60, cmap='bone')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(L_x1 ,L_best_result, z, label='(X1, Yminimum)')
ax.scatter(L_x2 ,L_best_result, z, label='(X2, Yminimum)')
ax.legend()
plt.show()

# Plot fitness terbaik di tiap generasi
plt.figure()
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Best Fitness in Each Generation')
plt.plot(L_best_fitness)
plt.show()

# Plot hasil nilai minimum terbaik di tiap generasi
plt.figure()
plt.xlabel('Generations')
plt.ylabel('Minimum Value')
plt.title('Best Result in Each Generation')
plt.plot(L_best_result)
plt.show()