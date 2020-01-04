# CNN-in-systemC-


## Step
1. file input 28\*28 pixel of image (need to divide to 256.0 to floating point).
	* load 5\*5 kernel (or called filter) data from ROM.
2. First Convolution
	* Stride: 1
	* 28\*28 
	* Create 24\*24\(matrix) \* 6 (channel) Convolved feature.
	* store(write) into RAM (index: 0~3455).
	* Adding bias.
	* Normalisation: using Activation function $ReLU(x) = max(x,0)$
3. Pooling
	* Stride: 2
	* Load First Convulution layer data (24\*24)\*6 from RAM.
	* Create 12\*12\*6 filter and store(write) into RAM (index: 3456~4319).
4. Second Convolution
	* load 5\*5\*6(depth) kernel (or called filter) data from ROM.
	* Stride: 1
	* Depth: 6
	* Create 8\*8 (matrix) \* 16 (Depth or called channel) Convolved feature.
	* Store(write) into RAM (index: 4320~5343)
	* Adding bias
	* Normalisation: using Activation function $ReLU(x) = max(x,0)$