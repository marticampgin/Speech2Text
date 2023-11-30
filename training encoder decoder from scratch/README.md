## Jupyter notebook used to train a Transformer-based encoder-decoder from scratch on the Norwegian Speech Commands Corpus 
This notebook contains the code that we utilized to train our own encoder-decoder model from scratch. If you want to run this notebook, make sure that you have installed all the packages from requirements.txt, your Python version is 3.11.5 and that you have 
```
- 'dataset.csv'-file in the same directory as the notebook
- 'wav_recordings'-folder that contains wav-recordings extracted from the corpus in the same directory as the notebook
-  an empty 'log_rectangular'-folder in the same directory as the notebook
-  set produce_spectr variable to True
-  change the patch & image variables, as well as resize-transfrom according to size of your spectrograms
-  spectrogram sizes can be changed by changing plt.figure(figsize(width, height)) width and height parameters
```
It is also possible that after generating spectrograms from .wav-files it will be neccessary to re-initialize the dataset while also changing produce_spectr variable back to False
