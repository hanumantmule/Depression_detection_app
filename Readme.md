# Depression Detection using BiLSTM and 1D CNN-Based Model
Depression is an international mental state downside, the worst of which may cause self-harm or suicide. So, automatic detection of depression is useful for the clinical diagnosing. We have designed a new depression detection technique that uses text transcripts and audio content from patient's interviews.

## Getting Started
Clone this project using below github url: https://github.com/hanumantmule/Depression_detection_app.git

### Prerequisites

Python packages you need to install. The list of libraries and its version is added in requirement.txt file. 

* [Flask](https://flask.palletsprojects.com/en/2.0.x/) - Flask is a web application framework written in Python.
* [NumPy](https://pypi.org/project/numpy/) - NumPy is an open-source numerical Python library.
* [pandas](https://pandas.pydata.org/) - Python Data Analysis Library.
* [matplotlib](https://matplotlib.org/) - Library for creating static, animated, and interactive visualizations.
* [seaborn](https://seaborn.pydata.org/) - Python data visualization library based on matplotlib.
* [nltk](https://www.nltk.org/) - Natural Language Toolkit 
* [sklearn](https://scikit-learn.org/) - Simple and efficient tools for predictive data analysis.
* [imblearn](https://pypi.org/project/imblearn/) - Imbalanced-learn is a python package offering a number of re-sampling techniques.
* [allennlp]() - NLP library for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
* [allennlp-models]() - Provides an easy way to download and use pre-trained models.
* [python_speech_features]() - This library provides common speech features for ASR including MFCCs and filterbank energies.
* [torch]() - PyTorch is a Python package that provides Tensor computation and Deep neural networks built.
* [wave]() -Read and write WAV files.
* [librosa]() - A python package for music and audio analysis.
* [tensorflow]() - Open source software library for high performance numerical computation.
* [re]() - A compendium of commonly-used regular expressions.
* [pyAudioAnalysis]() - Python library covering a wide range of audio analysis tasks.
* [eyed3]() - Python tool for working with audio files.
* [scipy]() - Scientific Library for Python

## Installing
Here we are using ```PIP``` which is a package manager for Python packages.

Note: If you have Python version 3.4 or later, PIP is included by default.
To install the all the dependencies for the project. Type below command in python console. 
```
pip install -r requirements.txt
```
This will install all the libraries mentioned in the requirements.txt file.

We have divided this application into **two** activities: 
1. **Model evaluation** and saving the model into the disk.

2. **Front end** using python web framework flask --> A folder named as 'Web App', contains the zip file of the web application code. 

**Note:** User need to save the model using the step 1 and then web application will use the saved model for prediction.
## Steps involved in model evaluation
1. Download the DAIC-WOZ Database : [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)
2. Extract the text and audio files from each folder.
3. Open ```Audio_CNN_Model.ipynb``` file and train the 1D CNN on the audio files. Save the model to disk.
4. Open ```bilstm.py``` file and train the BiLSTM model on the text transcript data.
5. Open ```fusion_net.ipynb``` file and train the multi-modal fusion by giving correct names of the 1D CNN and BiLSTM model saved in step 3 and 4. 

## Steps involved in web application setup
1. Copy the fusion model saved in above step 5, inside the 'data' folder.
2. Import the ```Web App``` code folder into the IDEs like ```pycharm and IntelliJ```.
3. Open the ```App``` folder and run the 'app.py' file.

## How to use ?

We have developed a web app which will provide depression detection service to the end user. User need to choose the audio and transcript file of the interview. Once user submit, as an output user will get the depression label and severity score.
This is snapshot of the home page of the application.

![Home Page](https://github.com/hanumantmule/Depression_detection_app/blob/master/Screenshots/home%20page.png?raw=true)

**Steps to use the web application.**

1. Launch the web application by running the ```app.py``` file
3. Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in the browser.
4. Select the audio and transcript file of the interview.
5. Click the submit button. 
6. Visualize the depression label and severity score.

## Results
For audio and text feature extraction we have used Mel spectrogram and Elmo respectively. We have analyzed the performance of the fusion model with above configuration. 

![Fusion Result](https://github.com/hanumantmule/Depression_detection_app/blob/master/Screenshots/accuracy.PNG?raw=true)

We have analyzed the performance of the BiLSTM model with respect to ```Elmo``` and ```BERT base``` embedding.

![Text Result](https://github.com/hanumantmule/Depression_detection_app/blob/master/Screenshots/text-exp-res.jpg?raw=true)

Similarly, For audio we have analyzed the 1D CNN model performance on ```MFCC``` and ```Mel Spectrogram```.

![Text Result](https://github.com/hanumantmule/Depression_detection_app/blob/master/Screenshots/audio-exp-res.jpg?raw=true)

## Future Scope
In addition to text and audio input, facial expressions in the form of video can be utilized for depression detection task.

## References
[1] https://pip.pypa.io/en/stable/reference/pip_install/  
[2] https://dcapswoz.ict.usc.edu/  
[3] https://www.mdpi.com/2076-3417/10/23/8701          
[4] https://pypi.org/project/Flask/
## Contributing

Please read [CONTRIBUTING.md](https://github.com/hanumantmule/Email_Classification/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Hanumant Mule** 
* **Namrata Kadam** 
* **Manisha Sharma** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

