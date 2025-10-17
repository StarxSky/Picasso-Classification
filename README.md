# Picasso-Classification
Picasso-Classification : a model to classification the abstract image from hand-drawn.

# How to use it?
* Step 1: install the requirements: (Terminal)
```
> sudo apt install git
> git https://www.github.com/starxsky/Picasso-Classification.git
> cd Picasso-Classification 
> pip3 install -r requirements.txt  
```

* Step 2: Click [here](https://github.com/StarxSky/Picasso-Classification/releases/download/v1.0/dataset.zip) to download the datasets and unzip to the current path:
When you has already, its looks like this:

```
Picasso-Classification/
----------------------
      |- CNN_train.py # CNN
      |- model.py
      |- dataset.py
      |- FCNN.py # Full-conection Net
      |- fcnet_predict.py
      |- cnn_predict.py
      |- dataset/
           | - train/
                ........
                ........
           | - test/
                ........
                ........
```

* Step 3: Training one of the models (CNN / FCNet) [Terminal]:
```
>python3 CNN.py 

```

* Step 4: Predict the test-image (if you want ) :
please keeps the type of model is same as your trained model.
```
>python3 cnn_predict.py
```

