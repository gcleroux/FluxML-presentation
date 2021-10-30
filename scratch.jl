### Scratch file to create a simple model to analyze Iris dataset

using CUDA

using Flux
using Flux: onehotbatch, onecold
using Flux.Data: DataLoader

using MLDatasets
using MLDataUtils

using ProgressMeter: @showprogress

function get_data(args)

    ENV["DATADEPS_ALWAYS_ACCEPT"] = true

    # Get the training dataset
    data = MLDatasets.Iris.features()
    labels = MLDatasets.Iris.labels()

    # Formatting the labels
    formatted_labels = Int64[]
    for label in labels
        if occursin("setosa", label)
            push!(formatted_labels, 0)

        elseif occursin("versicolor", label)
            push!(formatted_labels, 1)

        else
            push!(formatted_labels, 2)
        end
    end

    formatted_labels = onehotbatch(formatted_labels, 0:2)

    # Separating the datasets in training/testing
    train_data, test_data = stratifiedobs((data, formatted_labels), p=0.8)

    train_loader, test_loader = DataLoader(train_data, batchsize=args.bs, shuffle=true), DataLoader(test_data, batchsize=args.bs)

    return train_loader, test_loader

end

function create_model()
    
    # Creating a simple linear model
    model = Flux.Chain(
        Flux.Dense(4, 16, relu),
        Flux.Dense(16, 42, relu),
        Flux.Dense(42, 3)
    )

    return model
end

function get_device(args)
    # Choosing the device
    if CUDA.functional() && args.use_cuda
        CUDA.allowscalar(false)
        return gpu

    else
        return cpu
    end
end


struct args
    lr::Float64
    bs::Int
    epochs::Int
    use_cuda::Bool
end

hp = args(1e-2, 25, 15, true)

train_loader, test_loader = get_data(hp)

device = get_device(hp)

model = create_model() |> device

## Training the model
ps = Flux.params(model)

loss = Flux.Losses.logitcrossentropy

optimizer = Flux.ADAM(hp.lr)

for epoch in 1:hp.epochs
    for (index, (X, y)) in enumerate(train_loader)
        X, y = device(X), device(y)
        
        l = nothing
        gs = Flux.gradient(ps) do
            ŷ = model(X)
            l = loss(ŷ, y)
        end
        
        index == length(train_loader) ? println("Epoch: ", epoch, ", Loss: ", l) : nothing

        Flux.update!(optimizer, ps, gs)
    end
end

## Testing the model
acc = 0
n = 0

for (X, y) in test_loader

    X, y = device(X), device(y)

    ŷ = model(X)

    global acc += sum(onecold(ŷ) .== onecold(y))

    global n += size(X)[2]
end

println("\n\nAccuracy: ", acc/n)