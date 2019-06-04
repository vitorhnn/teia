using Distances

function kmeans(x, start, k)
    centroids = copy(start)#rand(Float32, k)
    distances = pairwise(Euclidean(), x, centroids, dims = 1)

    mins = findmin(distances, dims = 2)[2]

    for i ∈ 1:size(centroids, 1)
        ∑ = [0, 0]
        total = 0
        for row ∈ mins
            if row[2] == i
                ∑ += x[row[1], :]
                total += 1
            end
        end

        centroids[:, i] = ∑ ./ total
    end

    return centroids
end

x = [1 1; 1 2; 1 3]
start = [1.0 1.0; 1.0 2.0]
