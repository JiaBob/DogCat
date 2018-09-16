## Dog VS Cat

The program uses compare two methods on the DogCat dataset (modified, 20000 training, 4000 test). Both methods are based on CNN. One uses simple  two-convolution-layer CNN, the other one uses pretrained state-of-art model (ResNet). 

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

### Results

Training platform: AWS p2.xlarge (single k80 GPU)

Training duration: simple CNN (50 epochs) takes 23 mins; pretrained (fix feature layers) ResNet18 (10 epochs) takes 10 mins.

Accuracy: simple CNN gets 80% accuracy; pretrained ResNet18 gets 97.5% accuracy.



