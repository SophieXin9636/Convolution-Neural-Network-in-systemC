# CNN using in systemC

## Outline
[TOC]

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
3. First Pooling
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
5. Second Pooling
	* Stride: 2
	* Load First Convulution layer data (8\*8)\*16 from RAM. (index: 4320~5343)
	* Create 4\*4\*16 filter and store(write) into RAM (index: 5344~5599).
6. First fully connected layer
	* Load Second Pooling layer data from RAM. (index: 5344~5599)
	* Write 128 neuron into RAM (index 5600~5727)
7. Second fully connected layer
    * Load First fully connected layer neuron data from RAM. (index: 5600~5727)
    * Write 84 neuron into RAM (index 5728~5811)
8. Third fully connected layer
    * Load Second fully connected layer neuron data from RAM. (index: 5728~5811)
    * write into Result to Monitor block, and it will output result.


## System Block

![](https://i.imgur.com/ohN4Hib.png)


## 執行結果
* MODE = 1 (floating), File = 1

![](https://i.imgur.com/UJXZKVU.png)

* MODE = 1 (floating), File = 2

![](https://i.imgur.com/jm4zhL2.png)



* MODE = 2 (floating), File = 1

![](https://i.imgur.com/x9HMKzY.png)

* pct (PA)

![](https://i.imgur.com/hgZNd6A.png)
