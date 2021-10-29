### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 3f2f70cc-4b66-43b9-a4e8-f1b7bfbde649
begin
	
	import Pkg
	Pkg.activate(Base.current_project())
	
	# Needed to load the datasets correctly in the notebook
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
	
	using CUDA
	
	# Import the needed libraries
	using Flux
	using Flux.Data: DataLoader
	using Flux: onehotbatch, onecold, @epochs
	using Flux.Losses: logitcrossentropy
	
	using MLDatasets
	using Plots
	using PlutoUI
	
	using Printf
end

# ╔═╡ 2b86039e-ffb3-41a9-949d-c1d7c8d14a2a
md"""
Classifying hand-written digits with AI using FluxML
====================================================
Notebook written by : Guillaume Cléroux, Université de Sherbrooke

In this notebook, we will cover the basics of using Flux.jl to develop neural networks capable of doing simple tasks like identifying written numbers. This notebook assumes that you have a basic understanding of the Julia language and some notion of Machine Learning. It is mainly oriented towards people that are interested in looking at Flux.jl and it's capabilities.


### A short word on FluxML
###### *Relax! Flux is the ML library that doesn't make you tensor.* 
Flux is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.
"""

# ╔═╡ 949b109a-dee0-4d21-8a68-f372d9851f7c
md"""

Getting Started
---------------

First, we will import the needed libraries.


- Flux - The machine learning library that we are demonstrating in this notebook


- MLDatasets - The package we will use to get our data for training and testing. It contains a wide array of classic ML Datasets, feel free to explore them yourselves!


- Plots - A simple plotting library to visualize our data and plot our results.


- PlutoUI - A Pluto extension needed for the interactivity aspect of this notebook
"""

# ╔═╡ 445437f9-d6be-4345-966d-6158917651cf
md"""
Let's also create a data structure to store the hyper parameters that we want
"""

# ╔═╡ f5dd1c42-fd2e-4a94-b5ef-7f27c03417d5
# Creating the hyper parameters struct
mutable struct Hyper_Params
	lr::Float64
	batch_size::Int64
	epochs::Int64
	use_cuda::Bool
end

# ╔═╡ f1fb9547-aa68-48ed-954c-497877ef56f7
md"""
Now we can set our hyper parameters to any value we want and see how it affects the resulting model.
"""

# ╔═╡ 2d66a34e-a506-47fb-8f96-d9dd6dfbe247
hp = Hyper_Params(
	1e-3,	# Learning rate
	1000, 	# Batch size
	50,	# Nb epochs
	true 	# Training model on GPU
)

# ╔═╡ 689084da-b6b4-4eff-983f-1a8417804e7c
if CUDA.functional() && hp.use_cuda
	CUDA.allowscalar(false)
	device = gpu
	
else
	device = cpu
end

# ╔═╡ 10bca79d-aad4-4dae-a8c6-51d56766596c
md"""
## Having a first look at our data
It is always important to have a quick look at our data before tackling a problem. It allows us to familiarize ourselves with our data and might give us a better idea on how we will try to solve it.

Let's have a look at a few samples of data in the MNIST Dataset.
"""

# ╔═╡ 15bd08cb-d7fa-4221-a4dd-1e966919aad3
let
	# Creating a custom DataType to hold our grayscaled samples
	struct GrayscaledImageArray
		images::Array{Gray{Float32}, 3}
	end
	
	# Overloading the get method for simpler synthax when plotting
	Base.getindex(X::GrayscaledImageArray, i::Int64) = X.images[:, :, i]
	
	# Loading the training data samples
	train_x, train_y = MNIST.traindata(Float32, 1:9)
	
	# Converting samples to grayscale images
	imgs = GrayscaledImageArray([Gray.(i) for i in train_x])
	
	# Plotting the samples
	plot(plot(imgs[1]), plot(imgs[2]), plot(imgs[3]),
		 plot(imgs[4]), plot(imgs[5]), plot(imgs[6]),
		 plot(imgs[7]), plot(imgs[8]), plot(imgs[9]), layout=(3,3))
	
end

# ╔═╡ 3ecf19af-6979-4856-9524-2571f1b3d69e
md"""
Loading the datasets
--------------------
Now let's load the MNIST dataset using the MLDatasets package.

Doing so is very simple, we simply need to call the methods:
- MNIST.traindata()
- MNIST.testdata()

###### Training dataset
train\_x will represent the training images\
train\_y is the labels of the training images

###### Testing dataset
test\_x will represent the testing images\
test\_y is the labels of the testing images


Once we are done loading the datasets into variables, let's have a look at a few samples!
"""

# ╔═╡ 410a2642-6aa2-448a-af37-e7f9a296deb9
begin
	# Loading the training dataset
	train_x, train_y = MLDatasets.MNIST.traindata(Float32)
	
	# Reshape Data in order to flatten each image into a linear array
    train_x = reshape(train_x, 28, 28, 1, :)
	
    # One-hot-encode the labels
    train_y = onehotbatch(train_y, 0:9)
	
	nothing # hide the output
end

# ╔═╡ 82fb0174-bc96-4af9-9e8c-f769f6702921
begin
	# Loading Dataset	
    test_x, test_y = MLDatasets.MNIST.testdata(Float32)
	
	# Reshape Data in order to flatten each image into a linear array
    test_x = reshape(test_x, 28, 28, 1, :)

    # One-hot-encode the labels
	test_y = onehotbatch(test_y, 0:9)
	
	nothing
	
end

# ╔═╡ 1bbfac84-1827-4a77-a76d-9ad6ecac4c32
begin
    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((train_x, train_y), batchsize = hp.batch_size, shuffle = true)
    test_loader = DataLoader((test_x, test_y), batchsize = hp.batch_size)
	
	nothing # hide the output
end

# ╔═╡ 327b1d02-edbb-4f4f-8440-dedb3364fd74
md"""
Creating the model
==================
"""

# ╔═╡ 794cd5a0-a674-436f-b5f1-6e010f08037a
begin
	model = Flux.Chain(
		
		# First convolutionnal layer
		Flux.Conv((5, 5), 1 => 6, relu),
		Flux.MeanPool((2, 2), pad=1),
		
		# Second convolutional layer
		Flux.Conv((4, 4), 6 => 16, relu),
		Flux.MeanPool((2, 2), pad=1),
		
		# Flattening the output before giving it to the fully connected layer
		flatten,
		
		# Fully connected layer
		Flux.Dense(16*6*6, 120, relu),
		Flux.Dense(120, 69, relu),
		Flux.Dense(69, 10)
		)
	
	# Sending the model on the selected device
	model = model |> device
end

# ╔═╡ ef4fb132-7a6e-4841-b4d5-c74fca6647ef
loss = logitcrossentropy

# ╔═╡ 2672ca60-a466-4aa0-97bf-da15a14d01f8
params = Flux.params(model) # model's trainable parameters

# ╔═╡ 39f36135-bfed-4ef1-9438-f979963bbf32
optimizer = ADAM(hp.lr)

# ╔═╡ 80401829-106f-4f2f-92ff-fbbc1c8cf98c
md"""
### Training the model
"""

# ╔═╡ 46c7828e-98dd-444f-8130-4de791053b83
begin
	
	with_terminal() do
		
		println("Training the model\n")
		
		for epoch in 1:hp.epochs
			
			# Iterating through data_loader
			for (index, (x, y)) in enumerate(train_loader)
				
				# Sending the tensors to selected device
				x, y = device(x), device(y)
				
				l = nothing
				gradient = Flux.gradient(params) do
					ŷ = model(x)
					l = loss(ŷ, y)
				end
				
				if epoch % 5 == 0 && index == length(train_loader)
					@printf "Epoch: %d/%d, loss: %.4f\n" epoch hp.epochs l
				end
				
	
				Flux.Optimise.update!(optimizer, params, gradient)
			end
		end
	end
end

# ╔═╡ 8f93427c-f9c4-4de5-972c-6f3e962958c4
md"""
#### Showing the results
"""

# ╔═╡ a5494ded-78db-4994-94a7-82196d6ef689
@bind drawing HTML("""
<div id=parent>
	<canvas id=canvas width=112px height=112px></canvas>
	<button id=clearButton>clear</button>
</div>
	
<script>
	const canvasWidth = 112, canvasHeight = 112, background = "#f1f1f1";
	
	const parentDiv = currentScript.previousElementSibling
	const c = parentDiv.querySelector("canvas")
	const ctx = c.getContext("2d");
	ctx.fillStyle = background;
	ctx.fillRect(0, 0, canvasWidth, canvasHeight);
	
	let drawing = false;
	parentDiv.value = [];
	
	c.addEventListener('mousedown', () => drawing = true);
	c.addEventListener('mouseup', () => drawing = false);
	c.addEventListener('mousemove', (e) => {
		if(drawing) {
			ctx.beginPath();
			ctx.arc(e.offsetX, e.offsetY, 4, 0, 2 * Math.PI);
			ctx.fillStyle = "#010101";
			ctx.fill();
				
			parentDiv.value.push([e.offsetX, (canvasHeight - e.offsetY)]);
			parentDiv.dispatchEvent(new CustomEvent("input"));
		}
	});
	
	function clearCanvas(e) {
		ctx.fillStyle = background;
		ctx.fillRect(0, 0, canvasWidth, canvasHeight);
		parentDiv.value = [];
		parentDiv.dispatchEvent(new CustomEvent("input"));
	}
	
	parentDiv.querySelector("#clearButton").addEventListener('click', clearCanvas);
</script>
""")

# ╔═╡ b7cad210-7bbe-48ef-9ffa-89987a0b3e5c
function convert_drawing(d)
	
	# Creating a template matrix with padding
	mat = zeros(Float32, 116, 116, 1, 1)

	# Transforming drawing array into a matrix
	for p in d

		let
			x = p[1] + 2
			y = p[2] + 2

			mat[x, y, 1, 1] += 14.

			# Adding some noise around the points 
			mat[x-1, y, 1, 1] += 3.
			mat[x+1, y, 1, 1] += 3.
			mat[x, y-1, 1, 1] += 3.
			mat[x, y+1, 1, 1] += 3.
		end
	end

	# Reduce the matrix's size to 28x28 with a MeanPool
	downscaler = Flux.AdaptiveMeanPool((28, 28)) 

	return downscaler(mat)
end

# ╔═╡ ccb676ce-8c53-452b-a076-92e23f2e7049
### The plot is only here to visualize the drawing in a matrix of size 28x28
# It's here only for debugging and should be removed during the final revision

plot(Gray.(convert_drawing(drawing)[:, :, 1, 1]))

# ╔═╡ b4c3774b-c682-49cc-927a-c8525f462b20
begin
	
	# Sending the pred tensor to the cpu to allow scalar resizing
	pred = model(device(convert_drawing(drawing))) |> cpu
	
	pred = reshape(pred, 10)
	
	bar(pred, 
		title="Trained model's prediction", 
		label=nothing,
		xlims = (0, 11),
		xticks=(0:10, -1:9))
	
end

# ╔═╡ Cell order:
# ╟─2b86039e-ffb3-41a9-949d-c1d7c8d14a2a
# ╟─949b109a-dee0-4d21-8a68-f372d9851f7c
# ╠═3f2f70cc-4b66-43b9-a4e8-f1b7bfbde649
# ╟─445437f9-d6be-4345-966d-6158917651cf
# ╠═f5dd1c42-fd2e-4a94-b5ef-7f27c03417d5
# ╟─f1fb9547-aa68-48ed-954c-497877ef56f7
# ╠═2d66a34e-a506-47fb-8f96-d9dd6dfbe247
# ╠═689084da-b6b4-4eff-983f-1a8417804e7c
# ╟─10bca79d-aad4-4dae-a8c6-51d56766596c
# ╠═15bd08cb-d7fa-4221-a4dd-1e966919aad3
# ╟─3ecf19af-6979-4856-9524-2571f1b3d69e
# ╠═410a2642-6aa2-448a-af37-e7f9a296deb9
# ╠═82fb0174-bc96-4af9-9e8c-f769f6702921
# ╠═1bbfac84-1827-4a77-a76d-9ad6ecac4c32
# ╠═327b1d02-edbb-4f4f-8440-dedb3364fd74
# ╠═794cd5a0-a674-436f-b5f1-6e010f08037a
# ╠═ef4fb132-7a6e-4841-b4d5-c74fca6647ef
# ╠═2672ca60-a466-4aa0-97bf-da15a14d01f8
# ╠═39f36135-bfed-4ef1-9438-f979963bbf32
# ╟─80401829-106f-4f2f-92ff-fbbc1c8cf98c
# ╠═46c7828e-98dd-444f-8130-4de791053b83
# ╟─8f93427c-f9c4-4de5-972c-6f3e962958c4
# ╟─a5494ded-78db-4994-94a7-82196d6ef689
# ╠═b7cad210-7bbe-48ef-9ffa-89987a0b3e5c
# ╠═ccb676ce-8c53-452b-a076-92e23f2e7049
# ╠═b4c3774b-c682-49cc-927a-c8525f462b20
