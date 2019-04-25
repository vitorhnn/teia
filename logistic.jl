g(z) = 1 ./ (1 .+ ℯ.^(-z))

h(θ, x) = g(transpose(θ) * x)

J(θ, x, y, m) = -sum(y .* log.(h(θ, x)) .+ (1 .- y) .* log.(1 .- h(θ, x))) / m 

∇J(θ, x, y, m) = (transpose((h(θ, x) - y) * transpose(x))) ./ m

function descent(θ, α, ϵ, x, y)
    iterations = 0
    m = length(θ)
    
    while true
        iterations += 1
        nθ = θ - α * ∇J(θ, x, y, m)

        err = J(nθ, x, y, m)
        if err < ϵ || isnan(err)
            break
        end
        θ = nθ
    end

    θ, iterations, J(θ, x, y, m)
end