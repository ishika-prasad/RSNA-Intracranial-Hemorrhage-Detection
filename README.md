# RSNA-Intracranial-Hemorrhage-Detection

![RSNA-Intracranial-Hemorrhage-Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
Identify acute intracranial hemorrhage and its subtypes

This project was hosted on Kaggle platform as a competition. This codebase  uses InceptionV3 model for this classification task.

## Dataset

Dataset can be downlaoded from the kaggle website through following link : ![dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)


## Directory and code structure

<pre>
"Inception.py" files contains the main code to perform training, validation and testing 

"Preprocessing.py" contains the code for the image processing such as adjusting the window level for the MRI images

"Utils.py" contains helper function which are being used by ineption.py

</pre>

# How to run
<pre>
Just run "python inception.py" after setting the parameters in the inception.py and it would use the pretrained ImageNet weights and fine-tnue the model
</pre>
