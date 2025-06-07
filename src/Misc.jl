function NewtonRaphson(∇, H, x0, tol = 1e-6, max_iter = 100, nugget = 1e-6)
    """
    Newton-Raphson method for finding optimum of the function f with gradient ∇ and Hessian H.

    Inputs:
    ∇: Gradient function
    H: Hessian function
    x0: Initial guess
    tol: Tolerance for convergence
    max_iter: Maximum number of iterations

    Outputs:
    x: Approximate optimum
    """
    x = x0
    for i in 1:max_iter
        grad = ∇(x)
        if norm(grad) < tol
            return x
        end
        x -= inv(H(x) + nugget*I) * grad
    end
    error("Newton-Raphson did not converge within the maximum number of iterations")
end