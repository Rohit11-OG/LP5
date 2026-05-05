#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

int main() {
    int n = 1000000;

    vector<double> x(n), y(n);

    // Generate dummy data: y = 2x + 3
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = i * 0.001;
        y[i] = 2 * x[i] + 3;
    }

    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    // Parallel computation of sums
    #pragma omp parallel for reduction(+:sum_x,sum_y,sum_xy,sum_x2)
    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    // Calculate slope (m) and intercept (c)
    double m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double c = (sum_y - m * sum_x) / n;

    cout << "Slope (m): " << m << endl;
    cout << "Intercept (c): " << c << endl;

    return 0;
}