#include <omp.h>
#include <iostream>
#include <algorithm>
#include <numa.h>
#include <cmath>
#include <chrono>
#include <cstdlib>

struct Pivot {
    double val;
    int index;
};

struct drand48_data buffer;


// Comparison function
inline Pivot max_pivot(const Pivot& a, const Pivot& b) {
    if (a.val > b.val) {
        return a;
    } 
    return b;
}

#pragma omp declare reduction(maximum : struct Pivot : omp_out = max_pivot(omp_in, omp_out))

void generate_rand_matrix(int num_threads, int n, double** &A, double** &A_copy, double a, double b) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate memory for the array of pointers using numa_alloc_local
    A = reinterpret_cast<double**>(numa_alloc_local(n * sizeof(double*)));
    A_copy = reinterpret_cast<double**>(numa_alloc_local(n * sizeof(double*)));
    
    // Allocate and initialize each row of the matrices
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        A[i] = reinterpret_cast<double*>(numa_alloc_local(n * sizeof(double)));
        A_copy[i] = reinterpret_cast<double*>(numa_alloc_local(n * sizeof(double)));
    }

    // Fill the matrices with random numbers
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        drand48_data buffer_local; // Create a local buffer for each thread
        srand48_r(thread_id, &buffer_local); // Initialize the local buffer
        //srand48_r(thread_id, &buffer); // Initialize the random number generator with a seed

        for (int i = thread_id; i < n; i += num_threads) {
            for (int j = 0; j < n; j++) {
                double random_number;
                drand48_r(&buffer_local, &random_number); // Generate a random number
                random_number = a + (b - a) * random_number; // Scale the number to the [a, b) interval
                A[i][j] = random_number;
                A_copy[i][j] = random_number; // Direct copy
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Matrix generation runtime: " << duration.count() << " milliseconds" << std::endl;

}

double verification(int n, double** A, double** L, double** U, int* P) {
    auto start_time = std::chrono::high_resolution_clock::now();
    double res = 0.0;
    
    // Temporary matrix to store the result of L*U
    double** LU = new double*[n];
    for (int i = 0; i < n; ++i) {
        LU[i] = new double[n]{0.0};
    }

    // Calculating the LU product
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                LU[k][i] += L[k][j] * U[j][i];
            }
        }
    }
    
    // Calculating the residual by comparing A and LU after applying permutations from P
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            double diff = A[P[k]][i] - LU[k][i];
            res += std::pow(diff, 2);
        }
    }

    // Free memory
    for (int i = 0; i < n; ++i) {
        delete[] LU[i];
    }
    delete[] LU;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Verification runtime: " << duration.count() << " milliseconds" << std::endl;
    return res;
}


int decomposition(int n, double** A, double** L, double** U, int* P) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize permutation vector
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    // Allocate and initialize L, and U matrices using numa_alloc_local for cache optimization
    for (int i = 0; i < n; i++) {
        L[i] = reinterpret_cast<double*>(numa_alloc_local(n * sizeof(double)));
        L[i][i] = 1.0; // Ensuring L is a unit lower triangular matrix

        U[i] = reinterpret_cast<double*>(numa_alloc_local(n * sizeof(double)));
        for (int j = 0; j < n; j++) {
            U[i][j] = 0.0;
        }
    }

    int nworkers = omp_get_max_threads(); 
    int chunkSize = ((n - 1) / nworkers) + 1;
    
    // Create a copy of A for manipulation
    double** A_copy = new double*[n];
    #pragma omp parallel for schedule(static, chunkSize)
    for (int i = 0; i < n; i++) {
        A_copy[i] = new double[n];
        std::copy(A[i], A[i] + n, A_copy[i]);
    }

    Pivot* local_max_pivots = new Pivot[nworkers];

    #pragma omp parallel num_threads(nworkers)
    {
        int tid = omp_get_thread_num();
        local_max_pivots[tid].val = 0.0; // Initialize local max pivot value
        local_max_pivots[tid].index = -1; // Initialize local max pivot index
        
        for (int k = 0; k < n; k++) {
            #pragma omp for schedule(static, chunkSize)
            for (int i = 0; i < n; i++) {
                if (i >= k && std::fabs(A_copy[i][k]) > local_max_pivots[tid].val) {
                    local_max_pivots[tid].val = std::fabs(A_copy[i][k]);
                    local_max_pivots[tid].index = i;
                }
            }
            
            // Combine results: find global maximum among local maximums
            #pragma omp single
            {
                Pivot max_pivot = local_max_pivots[0];
                for (int i = 1; i < nworkers; i++) {
                    if (local_max_pivots[i].val > max_pivot.val) {
                        max_pivot = local_max_pivots[i];
                    }
                }
                
                // Apply permutation to P, A_copy, and L using the global maximum pivot
                std::swap(P[k], P[max_pivot.index]);
                std::swap(A_copy[k], A_copy[max_pivot.index]);
                for (int i = 0; i < k; i++) {
                    std::swap(L[k][i], L[max_pivot.index][i]);
                }
                for (int j = k; j < n; j++) {
                    U[k][j] = A_copy[k][j];
                }
                // Reset local max pivots for the next iteration
                for (int i = 0; i < nworkers; i++) {
                    local_max_pivots[i].val = 0.0;
                }
            }

            #pragma omp barrier

            #pragma omp for schedule(static, chunkSize)
            for (int i = k + 1; i < n; i++) {
                L[i][k] = A_copy[i][k] / U[k][k];
                for (int j = k; j < n; j++) {
                    A_copy[i][j] -= L[i][k] * U[k][j];
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Decomposition runtime: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads> [-v]" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int num_threads = std::atoi(argv[2]);
    bool verify = false;

    // Check if '-v' flag is passed for verification
    if (argc == 4 && std::string(argv[3]) == "-v") {
        verify = true;
    }

    if (n <= 0 || num_threads <= 0) {
        std::cerr << "Error: matrix_size and num_threads must be positive integers." << std::endl;
        return 1;
    }
    omp_set_num_threads(num_threads);

    // Allocate memory for matrices
    double** A = new double*[n];
    double** A_copy = new double*[n];
    double** L = new double*[n];
    double** U = new double*[n];
    int* P = new int[n];

    for(int i = 0; i < n; ++i) {
        A[i] = new double[n];
        A_copy[i] = new double[n];
        L[i] = new double[n]{0};
        U[i] = new double[n]{0};
    }

    double a = -10000;
    double b = 10000;
    generate_rand_matrix(num_threads, n, A, A_copy, a, b);

    if (decomposition(n, A, L, U, P) == -1) {
        std::cerr << "Matrix is singular and cannot be decomposed." << std::endl;
        return 1;
    }
    if (verify) {
        double res = verification(n, A_copy, L, U, P);
        std::cout << "Verification Result: " << res << std::endl;
    }

    return 0;
}
