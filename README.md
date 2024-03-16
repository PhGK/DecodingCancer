# 
Code base for [**Decoding pan-cancer treatment outcomes using multimodal real-world data and explainable AI**](https://www.medrxiv.org/content/10.1101/2023.10.12.23296873v2)

Install requirements can be found in requirements.txt.

To start the the analysis, run the Main file


```
python Main.py
```

Data will prepared in data.py and loaded. Here, this is a artificial dataset in which the outcome depends on a single variable.

The *crossvalidate* function will split data into five folds and start

- A pan cancer training, 
- A single entity cancer training
- LRP computation based on the trained model from the pan cancer training

Two new folders (*results* and *LRP*) are generated and contain the training and LRP results, respectively.




To change settings with respect to model architecture or training regimen, see settings_tt.py. By default, code is run on the gpu, this can be changed using the device variable in settings.






