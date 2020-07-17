# TF2-MNIST-AAE
Adversarial Autoencoder Implementation for MNIST handwritten digit in Tensorflow 2(Keras)

## Usage
### Command
```
python aae.py --ep <Number of Epochs> --batch <Batch Size>
```

### Arguments
* `--ep`: Number of Epochs to Train *Default*: `100`
* `--batch`: Batch Size *Default*: `100`

## Results
### Gaussian Prior Distribution
<table align='center'>
<tr align='center'>
<td> Generated Image (100 Epochs) </td>
<td> Distribution of Validation Data (100 Epochs) </td>
</tr>
<tr>
<td><img src = './img/grid_100_100.png' height = '400px'>
<td><img src = './img/classes_100_100.png' height = '400px'>
</tr>
</table>
