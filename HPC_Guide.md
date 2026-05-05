# HPC Practicals — Detailed Beginner Guide

---

## BASICS FIRST: What is HPC?

**HPC = High Performance Computing** — making programs run FASTER by using multiple CPU cores at the same time.

Your computer has multiple cores (usually 4-8). Normally, a program uses only 1 core. With HPC, we split work across ALL cores = faster execution.

## What is OpenMP?

- **OpenMP = Open Multi-Processing**
- A library for writing parallel programs in C/C++
- You add special comments called **pragmas** (`#pragma omp ...`)
- The compiler automatically splits the work across threads
- **Compile command:** `g++ -fopenmp filename.cpp -o output`

## What is a Thread?

A thread is a lightweight worker. If you have 4 cores, you can run 4 threads simultaneously.

```
Normal program:    Thread 1 does ALL the work
Parallel program:  Thread 1 does 25%, Thread 2 does 25%, Thread 3 does 25%, Thread 4 does 25%
                   → Finishes ~4x faster!
```

## Key OpenMP Commands (memorize these!)

| Command | What it does |
|---------|-------------|
| `#pragma omp parallel for` | Splits a for-loop across multiple threads |
| `#pragma omp critical` | Only ONE thread can enter this block at a time (like a lock) |
| `#pragma omp parallel sections` | Runs different code blocks on different threads |
| `#pragma omp section` | Marks one block inside `parallel sections` |
| `#pragma omp task` | Creates an independent job that any free thread can pick up |
| `#pragma omp single` | Only ONE thread runs this block; others skip it |
| `#pragma omp parallel for reduction(+:sum)` | Each thread gets private copy, combines results at the end |

## What is a Race Condition?

When two threads read/write the same variable at the same time → wrong results.

```
Example: sum = 0. Thread 1 reads sum=0, adds 5. Thread 2 reads sum=0, adds 3.
Thread 1 writes sum=5. Thread 2 writes sum=3.
Final sum = 3 (WRONG! Should be 8)
```

**Fix:** Use `#pragma omp critical` or `reduction`.

---
---

# PRACTICAL 1: Parallel BFS & DFS

## What are we doing?
Traversing (visiting all nodes of) a graph using BFS and DFS algorithms, but in **parallel** using OpenMP.

## What is a Graph?
A set of **vertices** (nodes) connected by **edges** (lines).

```
       0
      / \
     1   2
    / \   \
   3   4   5
```
- Vertices: 0, 1, 2, 3, 4, 5
- Edges: 0-1, 0-2, 1-3, 1-4, 2-5

## BFS vs DFS

| | BFS | DFS |
|---|---|---|
| Full form | Breadth-First Search | Depth-First Search |
| Strategy | Visit level by level (all neighbors first) | Go deep first, then backtrack |
| Uses | Queue (First In, First Out) | Recursion / Stack (Last In, First Out) |
| Output for above graph | 0 → 1, 2 → 3, 4, 5 | 0 → 1 → 3 → 4 → 2 → 5 |

---

## Line-by-Line Code Explanation

### Headers and Setup
```cpp
#include <iostream>
```
- For `cout` (printing) and `cin` (reading input)

```cpp
#include <vector>
```
- For `vector` — a resizable array

```cpp
#include <queue>
```
- For `queue` — FIFO data structure (used in BFS)

```cpp
#include <omp.h>
```
- OpenMP header — enables all `#pragma omp` directives

```cpp
using namespace std;
```
- So we can write `cout` instead of `std::cout`

---

### Graph Class
```cpp
class Graph {
    int V;
    vector<vector<int>> adj;
```
- `V` = number of vertices
- `adj` = adjacency list. `adj[0] = {1, 2}` means vertex 0 connects to 1 and 2

```cpp
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }
```
- Constructor: sets V and creates V empty lists

```cpp
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
```
- Adds edge between u and v (both directions because undirected graph)

---

### Parallel BFS

```cpp
void parallelBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
```
- `visited` = track which nodes we already visited (all start as false)
- `q` = queue for BFS

```cpp
    visited[start] = true;
    q.push(start);
```
- Mark starting node as visited, add it to queue

```cpp
    while (!q.empty()) {
        int size = q.size();
```
- Keep going until queue is empty
- `size` = number of nodes at current level

```cpp
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
```
- Process all nodes at this level IN PARALLEL
- If level has 3 nodes, 3 threads can work simultaneously

```cpp
            int node = -1;
            #pragma omp critical
            {
                if (!q.empty()) {
                    node = q.front();
                    q.pop();
                    cout << node << " ";
                }
            }
```
- `#pragma omp critical` = only ONE thread can do this at a time
- **Why?** Queue is shared — if two threads pop at same time, data corrupts
- Gets the front node, removes it from queue, prints it

```cpp
            if (node != -1) {
                for (int neighbor : adj[node]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
```
- For each neighbor of current node:
- Check if visited (inside critical — prevents two threads visiting same node)
- If not visited → mark it, add to queue

---

### Parallel DFS

```cpp
void parallelDFSUtil(int node, vector<bool>& visited) {
    bool alreadyVisited;
    #pragma omp critical
    {
        alreadyVisited = visited[node];
        if (!visited[node]) {
            visited[node] = true;
            cout << node << " ";
        }
    }
```
- Safely checks and marks the node as visited
- Uses critical section because `visited` is shared by all threads

```cpp
    if (alreadyVisited) return;
```
- If already visited, stop — don't go deeper

```cpp
    #pragma omp parallel for
    for (int i = 0; i < adj[node].size(); i++) {
        int neighbor = adj[node][i];
        if (!visited[neighbor]) {
            #pragma omp task
            parallelDFSUtil(neighbor, visited);
        }
    }
```
- For each unvisited neighbor, create a **task**
- `#pragma omp task` = creates an independent job
- Any free thread can pick up any task → branches explored in parallel

```cpp
void parallelDFS(int start) {
    vector<bool> visited(V, false);
    #pragma omp parallel
    {
        #pragma omp single
        parallelDFSUtil(start, visited);
    }
}
```
- `#pragma omp parallel` = creates a team of threads
- `#pragma omp single` = only ONE thread starts the first call
- **Why single?** Without it, ALL threads would start from root → wasted duplicate work
- After the first call, `#pragma omp task` distributes work

---

### Main Function
```cpp
int main() {
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;
    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter edges (u v):" << endl;
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "Enter starting vertex: ";
    cin >> start;

    g.parallelBFS(start);
    g.parallelDFS(start);
    return 0;
}
```

**Sample Input:**
```
6        ← vertices
5        ← edges
0 1      ← edge between 0 and 1
0 2
1 3
1 4
2 5
0        ← start from vertex 0
```

**Sample Output:**
```
Parallel BFS Traversal: 0 1 2 3 4 5
Parallel DFS Traversal: 0 1 3 4 2 5
```

---
---

# PRACTICAL 2: Parallel Bubble Sort & Merge Sort

## What are we doing?
Sorting an array using Bubble Sort and Merge Sort — both **sequential** (normal) and **parallel** (OpenMP) — then comparing their speed.

## What is Bubble Sort?
Repeatedly compare adjacent elements and swap if they're in wrong order.

```
Pass 1: [5,3,8,1] → [3,5,1,8] → largest element "bubbles" to the end
Pass 2: [3,5,1,8] → [3,1,5,8]
Pass 3: [3,1,5,8] → [1,3,5,8] ← sorted!
```

## What is Merge Sort?
Divide array in half, sort each half, then merge them back together.

```
[5,3,8,1] → split → [5,3] and [8,1]
[5,3] → split → [5] and [3] → merge → [3,5]
[8,1] → split → [8] and [1] → merge → [1,8]
[3,5] + [1,8] → merge → [1,3,5,8] ← sorted!
```

---

## Line-by-Line Code Explanation

### Setup
```cpp
#include <iostream>
#include <vector>
#include <cstdlib>   // for rand() — random number generation
#include <ctime>     // for time() — seed random numbers
#include <omp.h>     // for OpenMP
using namespace std;
#define SIZE 10000   // array size — 10,000 elements
```

---

### Sequential Bubble Sort
```cpp
void bubbleSortSeq(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```
- Outer loop: n-1 passes
- Inner loop: compare adjacent pairs, swap if left > right
- `n - i - 1`: last i elements are already sorted, skip them
- Time complexity: O(n²)

---

### Parallel Bubble Sort (Odd-Even Transposition) ⭐

**The problem:** In normal bubble sort, comparing (j, j+1) and (j+1, j+2) OVERLAP at index j+1 → can't do in parallel.

**The solution: Odd-Even approach**
- EVEN phase: compare pairs at indices (0,1), (2,3), (4,5) → NO overlap
- ODD phase: compare pairs at indices (1,2), (3,4), (5,6) → NO overlap

```cpp
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
```
- Outer loop: n passes needed for odd-even sort

```cpp
        // Even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
```
- `j += 2` = jump by 2 → picks indices 0, 2, 4, 6...
- Compares pairs: (0,1), (2,3), (4,5)... → no overlap!
- `#pragma omp parallel for` → each pair handled by a different thread

```cpp
        // Odd phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
```
- Starts at j=1 → picks indices 1, 3, 5, 7...
- Compares pairs: (1,2), (3,4), (5,6)... → no overlap!

**No `#pragma omp critical` needed!** Each thread works on completely different indices.

---

### Merge Function (used by both sequential and parallel)
```cpp
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);
```
- Creates two temporary arrays: left half and right half

```cpp
    int i = 0, j = 0, k = l;
    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) arr[k++] = left[i++];
        else arr[k++] = right[j++];
    }
```
- Compare elements from both halves, put smaller one into main array

```cpp
    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
```
- Copy any remaining elements

---

### Sequential Merge Sort
```cpp
void mergeSortSeq(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSortSeq(arr, l, m);       // sort left half
        mergeSortSeq(arr, m + 1, r);   // sort right half
        merge(arr, l, m, r);           // merge sorted halves
    }
}
```
- Recursively splits array in half until single elements, then merges back
- Time complexity: O(n log n)

---

### Parallel Merge Sort ⭐
```cpp
void mergeSortParallel(vector<int>& arr, int l, int r, int depth) {
    if (l < r) {
        int m = (l + r) / 2;
```

```cpp
        if (depth <= 0) {
            mergeSortSeq(arr, l, m);
            mergeSortSeq(arr, m + 1, r);
```
- **Depth limit!** When depth reaches 0, stop creating threads → go sequential
- **Why?** Creating threads has overhead. Too many tiny threads = slower than sequential

```cpp
        } else {
            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSortParallel(arr, l, m, depth - 1);

                #pragma omp section
                mergeSortParallel(arr, m + 1, r, depth - 1);
            }
        }
```
- `#pragma omp parallel sections` = run the two sections on DIFFERENT threads
- `#pragma omp section` = each section is one unit of work
- Left half sorted by Thread A, Right half sorted by Thread B → simultaneously!
- `depth - 1` = reduce depth at each level

```cpp
        merge(arr, l, m, r);
    }
}
```
- After both halves are sorted (in parallel), merge them together
- Merge itself is sequential (hard to parallelize merging)

---

### Main Function — Timing
```cpp
int main() {
    vector<int> arr(SIZE), temp;
    srand(time(0));
    generateRandom(arr);
```
- Creates array of 10,000 elements filled with random numbers

```cpp
    double start, end;

    temp = arr;
    start = omp_get_wtime();
    bubbleSortSeq(temp);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << (end - start) << " sec\n";
```
- `temp = arr` → copy the SAME unsorted array (fair comparison)
- `omp_get_wtime()` → high-precision wall-clock timer
- Measures how many seconds the sort takes

The same pattern repeats for parallel bubble, sequential merge, and parallel merge.

```cpp
    // Parallel Merge Sort called with depth=4
    mergeSortParallel(temp, 0, SIZE - 1, 4);
```
- `depth=4` → at most 2⁴ = 16 parallel threads

---
---

# PRACTICAL 3: Parallel Reduction (Min, Max, Sum, Average)

## What are we doing?
Finding the minimum, maximum, sum, and average of an array using **parallel reduction** with OpenMP.

## What is Reduction?
Combining all elements into one result (like adding all numbers to get sum). With OpenMP reduction, each thread computes a **partial result**, then all partial results are combined at the end.

```
Array: [1, 2, 3, 4, 5, 6, 7, 8]

Thread 1: works on [1,2,3,4] → partial sum = 10
Thread 2: works on [5,6,7,8] → partial sum = 26

Final: 10 + 26 = 36 ← automatically combined by OpenMP
```

---

## Line-by-Line Code Explanation

### Min Value
```cpp
int minval(int arr[], int n) {
    int minval = arr[0];
```
- Start by assuming first element is the minimum

```cpp
    #pragma omp parallel for reduction(min : minval)
    for (int i = 0; i < n; i++) {
        if (arr[i] < minval) minval = arr[i];
    }
```
- `reduction(min : minval)` means:
  1. Each thread gets its OWN private copy of `minval`
  2. Each thread finds minimum in its portion
  3. After the loop, OpenMP takes the MINIMUM of all thread results
- No race condition — each thread has its own copy!

```cpp
    return minval;
}
```

---

### Max Value
```cpp
int maxval(int arr[], int n) {
    int maxval = arr[0];
    #pragma omp parallel for reduction(max : maxval)
    for (int i = 0; i < n; i++) {
        if (arr[i] > maxval) maxval = arr[i];
    }
    return maxval;
}
```
- Same logic but uses `reduction(max : maxval)`
- Each thread finds max in its portion, then OpenMP takes the overall max

---

### Sum
```cpp
int sum(int arr[], int n) {
    int sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
```
- `reduction(+ : sum)`:
  1. Each thread gets private `sum = 0`
  2. Each adds its portion of array
  3. OpenMP adds all partial sums together

---

### Average
```cpp
double average(int arr[], int n) {
    return (double)sum(arr, n) / n;
}
```
- Calls the `sum` function (which is already parallel) and divides by n
- `(double)` = cast to decimal so we get a decimal answer, not integer

---

### Main
```cpp
int main() {
    int n = 5;
    int arr[] = {1, 2, 3, 4, 5};
    cout << "The minimum value is: " << minval(arr, n) << endl;   // 1
    cout << "The maximum value is: " << maxval(arr, n) << endl;   // 5
    cout << "The summation is: " << sum(arr, n) << endl;          // 15
    cout << "The average is: " << average(arr, n) << endl;        // 3
    return 0;
}
```

**Available reduction operators:**
- `+` (addition), `-` (subtraction), `*` (multiplication)
- `min` (minimum), `max` (maximum)
- `&` (bitwise AND), `|` (bitwise OR), `^` (bitwise XOR)

---
---

# PRACTICAL 4: Parallel Linear Regression (AIML)

## What are we doing?
Computing the best-fit line `y = mx + c` for a dataset of 1 million points, using OpenMP to speed up the calculation.

## What is Linear Regression?
Finding a straight line that best fits the data points.
- `m` = slope (how steep the line is)
- `c` = intercept (where the line crosses the y-axis)

## Formula
```
m = (n × Σxy - Σx × Σy) / (n × Σx² - (Σx)²)
c = (Σy - m × Σx) / n
```
Where Σ means "sum of all". We need to compute 4 sums: Σx, Σy, Σxy, Σx²

---

## Line-by-Line Code Explanation

### Setup
```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;
```

### Main Function
```cpp
int main() {
    int n = 1000000;
    vector<double> x(n), y(n);
```
- Creates two arrays of 1 million elements each

```cpp
    // Generate dummy data: y = 2x + 3
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = i * 0.001;
        y[i] = 2 * x[i] + 3;
    }
```
- Generates test data where we KNOW the answer: slope=2, intercept=3
- `#pragma omp parallel for` → even data generation is parallel (1M points!)
- x values: 0, 0.001, 0.002, 0.003, ...
- y values: 3, 3.002, 3.004, 3.006, ... (following y = 2x + 3)

```cpp
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
```
- Initialize the 4 sums we need to compute

```cpp
    #pragma omp parallel for reduction(+:sum_x, sum_y, sum_xy, sum_x2)
    for (int i = 0; i < n; i++) {
        sum_x  += x[i];
        sum_y  += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }
```
- **This is the key line!**
- `reduction(+:sum_x, sum_y, sum_xy, sum_x2)`:
  - Each thread gets PRIVATE copies of all 4 variables (initialized to 0)
  - Each thread adds its portion of the array
  - After the loop, OpenMP adds all partial sums together
- **4 reductions in ONE loop** — very efficient!
- Without `reduction`, all threads would write to the same variable → race condition → wrong answer

```cpp
    double m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double c = (sum_y - m * sum_x) / n;
```
- Applies the linear regression formula
- This part is sequential (just 2 calculations, no need to parallelize)

```cpp
    cout << "Slope (m): " << m << endl;
    cout << "Intercept (c): " << c << endl;
    return 0;
}
```
- Should print: Slope (m) ≈ 2, Intercept (c) ≈ 3
- Matches our original equation y = 2x + 3 ✅

---

**Compile & Run:**
```bash
g++ -fopenmp aiml.cpp -o aiml
./aiml
```

**Output:**
```
Slope (m): 2
Intercept (c): 3
```
