using LinearAlgebra;

g(z) = 1 ./ (1 .+ ℯ.^(-z))

h(θ, x) = g(transpose(θ) * x)

J(θ, x, y, m) = -sum(-y .* log.(h(θ, x)) .- ((-y) .* log.(1 .- h(θ, x)))) / m

∇J(θ, x, y, m) = (transpose((h(θ, x) - y) * transpose(x))) ./ m

function descent(θ, α, ϵ, x, y)
    iterations = 0
    m = length(θ)
    converged = false

    while !converged
        iterations += 1
        θ = θ - α * ∇J(θ, x, y)

        if (J(θ, x, y, m) < ϵ)
            converged = true
        end
    end

    θ, iterations
end

