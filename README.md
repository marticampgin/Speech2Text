### This project involved creating a Speech2Text model that instead of relying on direct signal processing utilizes a Visual Transformer Encoder to extract features from a spectrogram. The features are then fed to a decoder (LSTM) that converts the extracted features to a speech.

The project was done together with my two coursemates from the University. My objective was to code the architecture, as well as connect everything into one working pipeline. Thus, I am responsible for most of the coding. 
We utilized the Norwegian Newspaper Corpus to train our model and conduct the experiments: https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-4/.

While conducting several experiments, the most comprehensive and fruitful is provided in the folder training w. pre-trained ViT (.py-files)
Instructions on how to repeat each of the experiments are provided in each separate folder. Make sure to download the dataset and extract a folder containing WAV files with recordings to repeat experiments. The dataset is not included in the repo due to its size (~2 GB). 

A more detailed and thorough description of the project and experiments is provided in the [PDF paper](paper.pdf). 
