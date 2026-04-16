#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <memory>

#include "lib/tinyexpr.h"

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;
using SystemFunction = std::function<Vector(const Vector &)>;

// Thresholds and safety limits for Newton's method
constexpr int MAX_ITER_HARD_CAP = 1000;
constexpr int STAGNATION_WINDOW = 5;
constexpr double DIVERGENCE_RATIO = 1e8;
constexpr double STAGNATION_EPS = 1e-14;
constexpr double EPSILON = 1e-12; // Tolerance for singular matrix detection
constexpr double JACOBIAN_STEP = 1e-8;

// Represents the final state of the Newton iterative solver
enum class solver_status
{
    CONVERGED,
    STAGNATION_DETECTED,
    DIVERGENCE_DETECTED,
    SINGULAR_JACOBIAN,
    ITERATION_BUDGET_EXHAUSTED
};

// Stores the complete output of the solver
struct solver_result
{
    solver_status status;
    int iterations_taken;
    Vector X;
    Vector residuals;
    double max_res;
};

// Computes the maximum absolute value in a vector (infinity norm)
static double compute_max_residual(const Vector &v)
{
    double max_val = 0.0;
    for (double val : v)
    {
        max_val = std::max(max_val, std::abs(val));
    }
    return max_val;
}

// Function to solve linear system J * delta = -F using Gaussian elimination
static Vector solve_linear_system(Matrix A, Vector b)
{
    int n = A.size();

    for (int i = 0; i < n; i++)
    {
        // Partial pivoting
        int max_row = i;
        for (int k = i + 1; k < n; k++)
        {
            if (std::abs(A[k][i]) > std::abs(A[max_row][i]))
            {
                max_row = k;
            }
        }

        // Check for singularity
        if (std::abs(A[max_row][i]) < EPSILON)
        {
            throw std::runtime_error("Singular matrix");
        }

        std::swap(A[i], A[max_row]);
        std::swap(b[i], b[max_row]);

        // Forward elimination
        for (int k = i + 1; k < n; k++)
        {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++)
            {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    Vector x(n);
    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }

    return x;
}

// Computes the Jacobian matrix numerically using finite differences
static Matrix numerical_jacobian(const SystemFunction &f, const Vector &x)
{
    int n = x.size();
    Vector fx = f(x);
    Matrix J(n, Vector(n, 0.0));

    for (int j = 0; j < n; ++j)
    {
        Vector x_step = x;
        x_step[j] += JACOBIAN_STEP;

        Vector fx_step = f(x_step);

        for (int i = 0; i < n; ++i)
        {
            J[i][j] = (fx_step[i] - fx[i]) / JACOBIAN_STEP;
        }
    }
    return J;
}

// Core Newton solver with limits, stagnation checks, and divergence detection
static solver_result solve_newton_system(const SystemFunction &f, const Vector &guess, double target_tol)
{
    solver_result res;
    res.X = guess;
    int n = guess.size();

    Vector f_val = f(res.X);
    double initial_max = compute_max_residual(f_val);
    double prev_max = initial_max;
    int stagnation_count = 0;

    for (int iter = 0; iter < MAX_ITER_HARD_CAP; ++iter)
    {
        f_val = f(res.X);
        double max_res = compute_max_residual(f_val);

        // 1. Check for convergence
        if (max_res <= target_tol)
        {
            res.status = solver_status::CONVERGED;
            res.iterations_taken = iter;
            res.residuals = f_val;
            for (auto &r : res.residuals)
                r = std::abs(r);
            res.max_res = max_res;
            return res;
        }

        // 2. Check for stagnation
        double improvement = prev_max - max_res;
        if (std::abs(improvement) < STAGNATION_EPS)
            stagnation_count++;
        else
            stagnation_count = 0;

        if (stagnation_count >= STAGNATION_WINDOW)
        {
            res.status = solver_status::STAGNATION_DETECTED;
            res.iterations_taken = iter;
            res.residuals = f_val;
            for (auto &r : res.residuals)
                r = std::abs(r);
            res.max_res = max_res;
            return res;
        }

        // 3. Check for divergence
        if (iter > 5 && initial_max > 0 && max_res > initial_max * DIVERGENCE_RATIO)
        {
            res.status = solver_status::DIVERGENCE_DETECTED;
            res.iterations_taken = iter;
            res.residuals = f_val;
            for (auto &r : res.residuals)
                r = std::abs(r);
            res.max_res = max_res;
            return res;
        }

        prev_max = max_res;

        // Calculate Jacobian and set up system J * delta = -F
        Matrix J_val = numerical_jacobian(f, res.X);
        Vector neg_f_val(n);
        for (int i = 0; i < n; ++i)
        {
            neg_f_val[i] = -f_val[i];
        }

        // Solve for increments (deltas)
        try
        {
            Vector delta = solve_linear_system(J_val, neg_f_val);
            for (int i = 0; i < n; ++i)
            {
                res.X[i] += delta[i];
            }
        }
        catch (const std::runtime_error &e)
        {
            res.status = solver_status::SINGULAR_JACOBIAN;
            res.iterations_taken = iter;
            res.residuals = f_val;
            for (auto &r : res.residuals)
                r = std::abs(r);
            res.max_res = max_res;
            return res;
        }
    }

    res.status = solver_status::ITERATION_BUDGET_EXHAUSTED;
    res.iterations_taken = MAX_ITER_HARD_CAP;
    res.residuals = f(res.X);
    for (auto &r : res.residuals)
        r = std::abs(r);
    res.max_res = compute_max_residual(res.residuals);
    return res;
}

// Output helper for the final algorithm status
static void print_solver_status(solver_status code, int iter)
{
    switch (code)
    {
    case solver_status::CONVERGED:
        std::cout << "\033[32m"
                  << "\nConverged successfully at iteration " << iter << ".\n"
                  << "\033[0m";
        break;
    case solver_status::STAGNATION_DETECTED:
        std::cerr << "\033[31m"
                  << "\nStagnation detected at iteration " << iter << ".\n"
                  << "Algorithm stopped improving the solution.\n"
                  << "\033[0m";
        break;
    case solver_status::DIVERGENCE_DETECTED:
        std::cerr << "\033[31m"
                  << "\nDivergence detected at iteration " << iter << ".\n"
                  << "The error is growing excessively. Try a different initial guess.\n"
                  << "\033[0m";
        break;
    case solver_status::SINGULAR_JACOBIAN:
        std::cerr << "\033[31m"
                  << "\nSingular (or near-zero) Jacobian encountered at iteration " << iter << ".\n"
                  << "System cannot be solved further from this point.\n"
                  << "\033[0m";
        break;
    case solver_status::ITERATION_BUDGET_EXHAUSTED:
        std::cerr << "\033[33m"
                  << "\nIteration budget exhausted (" << iter << " iterations).\n"
                  << "Solution may not be fully accurate.\n"
                  << "\033[0m";
        break;
    }
}

// Output helper for the roots and residuals
static void print_results(const solver_result &res, const std::vector<std::string> &var_names)
{
    std::cout << "\nResults:\n";
    for (size_t i = 0; i < res.X.size(); i++)
    {
        std::string v_name = (i < var_names.size()) ? var_names[i] : "var" + std::to_string(i);
        std::cout << v_name << " = " << std::fixed << std::setprecision(8) << res.X[i]
                  << "\033[90m"
                  << "  (|f|: " << std::scientific << res.residuals[i] << ")\n"
                  << "\033[0m";
    }
    std::cout << "\033[90m"
              << "\nmax|f| = " << res.max_res << "\n"
              << "\033[0m";
}

static void print_expression_error(const std::string &expr, int err)
{
    std::string spaces(err - 1, ' ');
    std::cerr << "\033[31m"
              << "\nAn error occurred while parsing expression\n"
              << expr << "\n"
              << spaces << "^"
              << "\033[0m\n";
}

// Parses multiple expressions and binds them to tinyexpr variables
static SystemFunction build_dynamic_system(const std::vector<std::string> &expr_strs, int n, std::vector<std::string> &out_var_names, int &err_idx, int &err_pos)
{
    // Shared state to hold variable values during evaluation
    auto vars_state = std::make_shared<std::vector<double>>(n, 0.0);

    // Create standard names (x, y, z) for 3 vars, or x1, x2... for more
    std::vector<te_variable> te_vars;
    out_var_names.clear();

    for (int i = 0; i < n; ++i)
    {
        if (i == 0)
            out_var_names.push_back("x");
        else if (i == 1)
            out_var_names.push_back("y");
        else if (i == 2)
            out_var_names.push_back("z");
        else
            out_var_names.push_back("x" + std::to_string(i + 1));

        // Bind the memory address of our shared state to tinyexpr
        te_vars.push_back({out_var_names.back().c_str(), &(*vars_state)[i], TE_VARIABLE, nullptr});
    }

    std::vector<std::shared_ptr<te_expr>> compiled_exprs;
    err_idx = -1;

    // Compile each expression
    for (int i = 0; i < n; ++i)
    {
        int err = 0;
        te_expr *e_raw = te_compile(expr_strs[i].c_str(), te_vars.data(), te_vars.size(), &err);

        if (err)
        {
            err_idx = i;
            err_pos = err;
            return {};
        }

        compiled_exprs.push_back(std::shared_ptr<te_expr>(e_raw, [](te_expr *ptr)
                                                          { te_free(ptr); }));
    }

    // Return the capturing lambda that acts as our SystemFunction
    return [compiled_exprs, vars_state](const Vector &x) -> Vector
    {
        Vector result(x.size());

        // Update the bound variables before evaluation
        for (size_t i = 0; i < x.size(); ++i)
        {
            (*vars_state)[i] = x[i];
        }

        // Evaluate each tinyexpr expression
        for (size_t i = 0; i < compiled_exprs.size(); ++i)
        {
            result[i] = te_eval(compiled_exprs[i].get());
        }
        return result;
    };
}

int main()
{
    int n;
    std::cout << "Enter number of variables in the system:\n> ";
    if (!(std::cin >> n) || n <= 0)
    {
        std::cerr << "Invalid number of variables.\n";
        return 1;
    }

    // Clear input buffer before getline
    std::cin.ignore(10000, '\n');

    std::vector<std::string> expressions(n);
    std::cout << "\nEnter " << n << " expressions (Variables: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << ((i == 0) ? "x" : (i == 1) ? ", y"
                                   : (i == 2)   ? ", z"
                                                : ", x" + std::to_string(i + 1));
    }
    std::cout << ")\n";

    for (int i = 0; i < n; i++)
    {
        std::cout << "Eq " << i + 1 << ": ";
        std::getline(std::cin, expressions[i]);
    }

    int err_idx = -1, err_pos = 0;
    std::vector<std::string> var_names;

    auto target_function = build_dynamic_system(expressions, n, var_names, err_idx, err_pos);

    if (err_idx != -1)
    {
        print_expression_error(expressions[err_idx], err_pos);
        return 1;
    }

    Vector guess(n);
    std::cout << "\nEnter " << n << " initial guess values (separated by space):\n> ";
    for (int i = 0; i < n; i++)
    {
        std::cin >> guess[i];
    }

    double tol;
    std::cout << "Enter target tolerance (e.g. 1e-6):\n> ";
    std::cin >> tol;

    std::cout << "\nStarting Newton solver...\n";

    solver_result res = solve_newton_system(target_function, guess, tol);

    print_solver_status(res.status, res.iterations_taken);

    if (res.status == solver_status::CONVERGED || res.status == solver_status::ITERATION_BUDGET_EXHAUSTED || res.status == solver_status::STAGNATION_DETECTED)
    {
        print_results(res, var_names);
    }

    return (res.status == solver_status::CONVERGED) ? 0 : 1;
}
