## This folder contains .py files needed to train the model and reaplicate our results. 

In order to run everything without any errors:
```
- make sure to install all needed packages by using pip install -r requirements.txt
- for some reason, on educloud the requirements doesn't always install all the packages at once, some of them might need to be installed by manually typing pip install       <package name>
- Python version should be == 3.11.5
- have a folder with .wav recordings from the corpus the path to which you provide as an --wav_folder argument 
- have an empty folder that the path to which you provide as an --spec-folder argument, that's where the spectrograms will be stored after conversion
- when running for the first time, set --wav2spec argument to True, since you won't have any spectrograms at first. this will convert .wav recordings to log-mel grayscale spectrograms. after you have done it, always run without this argument.
```

# Example:
----------
First run:
` python main.py --wav2vec True --wav_folder 'wav_recordings' --'spec_folder' 'spectrograms' `

Second run:
` python main.py --wav_folder 'wav_recordings' --'spec_folder' 'spectrograms' `
