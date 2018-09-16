## Dog VS Cat

The program uses compare two methods on the DogCat dataset. Both methods are based on CNN. One uses simple  two-convolution-layer CNN, the other one uses pretrained state-of-art model (ResNet). 

The model is implemented by PyTorch with tensorboard for visualization. The only reason I put it on GitHub is that I have to use AWS instance to train it.

Run command below to check the parameter:

```shell
python transfer.py --help
```

Run command below to open up tensorboard:

```shell
tensorboard --logdir log
```

Finished models are stored in "./model". 

 



