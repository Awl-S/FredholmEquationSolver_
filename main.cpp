#include <iostream>
#include <Eigen/Dense>
#include <fstream>

// Функция f(x) определена заранее
double f(double x) {
    return x * x;
}

// Функция ядра k(x, y)
double k(double x, double y) {
    return x * y;
}

class FredholmEquationSolver {
private:
    int N;
    double a, b;
    Eigen::VectorXd X, Y;
    Eigen::MatrixXd K;

public:
    // Конструктор для уравнения второго рода
    FredholmEquationSolver(int N, double a, double b) : N(N), a(a), b(b), X(N), Y(N), K(N, N) {
        fillKernelAndSolveSecondKind();
    }

    // Конструктор для уравнения первого рода
    FredholmEquationSolver(int N, double a, double b, bool firstKind) : N(N), a(a), b(b), X(N), Y(N), K(N, N) {
        if (firstKind) {
            fillKernelAndSolveFirstKind();
        }
    }

    void fillKernelAndSolveSecondKind() {
        // Заполняем вектор узлов
        for (int i = 0; i < N; ++i) {
            X(i) = a + i * (b - a) / (N - 1);
            Y(i) = f(X(i));
        }

        // Заполняем матрицу ядра
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                K(i, j) = k(X(i), X(j));
            }
        }

        // Решаем уравнение Фредгольма второго рода
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);
        Y = (I - K).lu().solve(Y);
    }

    void fillKernelAndSolveFirstKind() {
        // Заполняем вектор узлов
        for (int i = 0; i < N; ++i) {
            X(i) = a + i * (b - a) / (N - 1);
            Y(i) = f(X(i));
        }

        // Заполняем матрицу ядра и решаем уравнение Фредгольма первого рода
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                K(i, j) = k(X(i), X(j));
            }
            Y(i) = K.row(i).mean() * (b - a);  // Метод трапеций
        }
    }

    // Получение результата
    Eigen::VectorXd getSolution() {
        return Y;
    }
};

int main() {
    FredholmEquationSolver solver2ndKind(100, 0, 1);
    Eigen::VectorXd Y2ndKind = solver2ndKind.getSolution();

    std::ofstream file2ndKind("solution_2nd_kind.txt");
    if (file2ndKind.is_open()) {
        for (int i = 0; i < Y2ndKind.size(); ++i) {
            file2ndKind << "y2ndKind(" << i << ") = " << Y2ndKind(i) << std::endl;
        }
        file2ndKind.close();
    } else {
        std::cout << "Unable to open file for 2nd kind solution.\n";
    }

    FredholmEquationSolver solver1stKind(100, 0, 1, true);
    Eigen::VectorXd Y1stKind = solver1stKind.getSolution();

    std::ofstream file1stKind("solution_1st_kind.txt");
    if (file1stKind.is_open()) {
        for (int i = 0; i < Y1stKind.size(); ++i) {
            file1stKind << "y1stKind(" << i << ") = " << Y1stKind(i) << std::endl;
        }
        file1stKind.close();
    } else {
        std::cout << "Unable to open file for 1st kind solution.\n";
    }

    return 0;
}

