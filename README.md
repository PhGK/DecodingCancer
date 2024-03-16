# 
Code base for [**Decoding pan-cancer treatment outcomes using multimodal real-world data and explainable AI**](https://www.medrxiv.org/content/10.1101/2023.10.12.23296873v2)


To the the analysis, run the Main file


```
python Main.py
```

Data will prepared in data.py and loaded. Here, this is a artificial dataset in which the outcome depends on a single variable.

The *crossvalidate* function will split data into five folds and start

- A pan cancer training, 
- A single entity cancer training
- LRP computation based on the trained model from the pan cancer training

A new folder is generated containing all results.




