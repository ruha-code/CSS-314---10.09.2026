# CSS-314: Parallel Computing — Class 0 Diagnostic
# Student ID: 230103026 | Group: 06-P

print("=== PART A: CODE & TRACING ===")

# Task 1: Trace the code
print("\n--- Task 1 ---")
x = 5
y = 2
x = x + y
y = x * 2
x = y - x
print("x =", x)
print("y =", y)


# Task 2: Fix the bug
print("\n--- Task 2 ---")
numbers = [10, 20, 30, 40, 50]
total = 0

for i in range(len(numbers)):
    total = total + numbers[i]

print("Total sum =", total)


# Task 3: Nested loops
print("\n--- Task 3 ---")
count = 0
for i in range(10):
    for j in range(10):
        count = count + 1

print("Loop (a) executes:", count, "times")
print("Loop (b) executes:", 1000 * 1000, "times")
print("Loop (c) executes:", 10 * 1000000, "times")


print("\n=== PART B: ALGORITHMS & COMPLEXITY ===")

# Task 4: Compare Algorithm A and B
print("\n--- Task 4 ---")
print("Algorithm A work = 1,000,000 operations")
print("Algorithm B work = 1,000,000,000,000 operations")
print("Algorithm B grows faster.")


# Task 5: Complexity ranking
print("\n--- Task 5 ---")
print("Ranking from fastest-growing to slowest-growing:")
print("2^n, n^2, n log2 n, n, log2 n, 1")


# Task 6: Search for number 45
print("\n--- Task 6 ---")
my_list = [3, 8, 12, 17, 24, 31, 45, 51, 63]

# 1. Linear search
linear_index = -1
for i in range(len(my_list)):
    if my_list[i] == 45:
        linear_index = i
        break

# 2. Binary search
left = 0
right = len(my_list) - 1
binary_index = -1

while left <= right:
    mid = (left + right) // 2
    if my_list[mid] == 45:
        binary_index = mid
        break
    elif my_list[mid] < 45:
        left = mid + 1
    else:
        right = mid - 1

print("Linear search index:", linear_index)
print("Binary search index:", binary_index)


print("\n=== PART C: DATA & PROBLEM SOLVING ===")

# Task 7: Find maximum in dataset
print("\n--- Task 7 ---")
numbers_data = [100, 500, 20, 9999, 450]
max_number = numbers_data[0]

for i in range(len(numbers_data)):
    if numbers_data[i] > max_number:
        max_number = numbers_data[i]

print("Maximum number =", max_number)


# Task 8: Count letters
print("\n--- Task 8 ---")
letters = ["A", "B", "A", "C", "B", "A", "D", "C", "A", "B"]
counts = {}

for item in letters:
    if item in counts:
        counts[item] = counts[item] + 1
    else:
        counts[item] = 1

print("Counts =", counts)


# Task 9: Matrix operations
print("\n--- Task 9 ---")
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

# Matrix Addition (A + B)
add_result = [[0, 0], [0, 0]]
for i in range(2):
    for j in range(2):
        add_result[i][j] = A[i][j] + B[i][j]

# Matrix Multiplication (A * B)
mult_result = [[0, 0], [0, 0]]
for i in range(2):
    for j in range(2):
        for k in range(2):
            mult_result[i][j] = mult_result[i][j] + A[i][k] * B[k][j]

print("A + B =", add_result)
print("A * B =", mult_result)
print("Scalar multiplications = 8")
