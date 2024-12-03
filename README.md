# Retinal Disease Classification
---

Retinal eye diseases, such as diabetic retinopathy, macular degeneration, and glaucoma are some of the leading causes of vision loss worldwide. Early and accurate detection of these conditions is critical to preventing irreversible damage and to initiate treatment. However, traditional diagnostic methods that rely on manual analysis by optometrists or ophthalmologists can be time-consuming, subjective, or inaccessible to underserved populations.

This project aims to develop an image classification model to identify some of these retinal diseases from fundus photography. The primary focus is on maximizing sensitivity, as missing a diagnosis could lead to severe consequences for patients, including delayed treatment and potential vision loss. Achieving high sensitivity ensures that cases with potential disease are flagged for further review and additional diagnostic testing methods to minimize the risk of overlooked conditions.

Some challenges include:
* Variability in image quality, from different skills in the technician taking the photo, different quality machines, and interfering factors like dust on the imaging lens.
* Differentiating between subtle differences between similar diseases.
* Addressing class imbalance where some diseases are underrepresented.

The goal is to create an additional tool to aid optometrists/ophthalmologists in improving diagnostic accuracy and prioritizing patient health through minimal false negatives. To consider a model production ready I would need accuracy above 90% and sensitivity above 95%.


All image data and the final saved Keras model for this project is hosted on this [Google Drive](https://drive.google.com/drive/folders/1sbu1XlEluZJrbUmGFn5uPuOlhPmZ2QVV?usp=drive_link) in zip files for those that would like to run through the code themselves.

---
## Introduction


---
## [Image Analysis](./code/01-eda.ipynb)



### Data Dictionary

All data is from Kaggle's [Ocular Disease Recognition Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).

|Feature|Type|Description|
|---|---|---|
|**ID**|*int*|Unique patient identifier|
|**Patient Age**|*int*|Age of patient|
|**Patient Sex**|*str*|Sex of patient|
|**Left-Fundus**|*str*|Filename for image of left eye for this patient|
|**Right-Fundus**|*str*|Filename for image of right eye for this patient|
|**Left-Diagnostic Keywords**|*str*|Diagnosis words for left eye fundus photo|
|**Right-Diagnostic Keywords**|*str*|Diagnosis words for right eye fundus photo|
|**N**|*int*|Normal fundus photo categorical column|
|**D**|*int*|Diabetes fundus photo categorical column|
|**G**|*int*|Glaucoma fundus photo categorical column|
|**C**|*int*|Cataract fundus photo categorical column|
|**A**|*int*|Age Related Macular Degeneration fundus photo categorical column|
|**H**|*int*|Hypertension fundus photo categorical column|
|**M**|*int*|Pathological Myopia fundus photo categorical column|
|**O**|*int*|Other diseases/abnormalities fundus photo categorical column|
|**filepath**|*str*|Filepath to the image for observation|
|**labels**|*str*|Diagnostic label for observation|
|**target**|*str*|Diagnostic target for observation|
|**filename**|*str*|Filename of image for observation|
|**image files**|*.jpg*|Individual fundus photos|

---
## [Modeling](./code/02_modeling.ipynb)



---
## Insights and Next Steps

