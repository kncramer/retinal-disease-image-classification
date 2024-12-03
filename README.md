# Retinal Disease Classification
---

Retinal eye diseases, such as diabetic retinopathy, macular degeneration, and glaucoma are some of the leading causes of vision loss worldwide. Early and accurate detection of these conditions is critical to preventing irreversible damage and to initiate treatment. However, traditional diagnostic methods that rely on manual analysis by optometrists or ophthalmologists can be time-consuming, subjective, or inaccessible to underserved populations.

This project aims to develop an image classification model to differentiate a healthy retina from a non-healthy retina. The primary focus is on maximizing sensitivity, as missing a diagnosis could lead to severe consequences for patients, including delayed treatment and potential vision loss. Achieving high sensitivity ensures that cases with potential disease are flagged for further review and additional diagnostic testing methods to minimize the risk of overlooked conditions.

Some challenges include:
* Variability in image quality, caused by different skills in the technician taking the photo, different quality machines, and interfering factors like dust on the imaging lens.
* Differentiating between subtle differences between similar diseases.
* Addressing class imbalance where some diseases are underrepresented.

The goal is to create an additional tool to aid optometrists/ophthalmologists in improving diagnostic accuracy and prioritizing patient health through minimal false negatives. To consider a model production ready I would want accuracy above 80% and sensitivity above 95%. 


All image data and the final saved Keras model for this project are hosted on this [Google Drive](https://drive.google.com/drive/folders/1sbu1XlEluZJrbUmGFn5uPuOlhPmZ2QVV?usp=drive_link) in zip files for those that would like to run through the code themselves.

---
## Introduction
There are several retinal diseases represented in this project:

* ***Diabetes*** - While a systemic condition, it can affect the small blood vessel located in the retina. Severe cases that aren't caught early can require laser surgery to repair and leave permanent vision loss.

<center><img src='./data/multiclass/diabetes/224_right.jpg' width = '300'></center>
<br>

* ***Glaucoma*** - Characteristically increases pressure inside the eye that stresses the optic nerve. Vision loss from glaucoma starts in a patients' peripheral vision so they appear asymptomatic in the early stages. There are several treatment options to prevent the progression of glaucoma, but any damage caused to the optic nerve is irreversible.

<center><img src='./data/multiclass/glaucoma/746_right.jpg' width = '300'></center>
<br>

* ***Cataract*** - The lens inside the eye grows clowdy and opaque, usually with age, and the impact on vision is gradual. Can be treated with cataract surgery where the lens in taken out and replaced with a lens implant. Cataracts don't affect the retina directly, but they impede the view of the retina to detect other conditions.

<center><img src='./data/multiclass/cataract/112_right.jpg' width = '300'></center>
<br>

* ***Age Related Macular Degeneration (AMD)*** - There are two major types, dry and wet. They both affect the macula which directly affects our central vision. Neither kind is curable, but monitoring is key because if dry AMD turns to wet, vision loss can be sudden and severe.

<center><img src='./data/multiclass/amd/213_right.jpg' width = '300'></center>
<br>

* ***Hypertension*** - Another systemic condition that affects our blood vessels. Often asymptomatic until damage is severe, early detection from yearly screenings is crucial to prevent damage to the retina.

<center><img src='./data/multiclass/hypertension/28_right.jpg' width = '300'></center>
<br>

* ***Pathological Myopia*** - When a patient's vision is short sighted enough that the back of the eye experiences degenerative changes. While untreatable, these retinal changes mean the patient is at higher risk for events like retinal detachments and they may want more frequent retinal monitoring.

<center><img src='./data/multiclass/myopia/401_right.jpg' width = '300'></center>
<br>

The annual eye exam is the forefront way to screen for these conditions and many others. The historical method is for an optometrist or ophthalmologist to perform a dilated fundus exam where they dilate the pupil and physically look at the back of the eye. The increasingly preferred method is to take a photo of the retina with a fundus photography camera. They are not available in all practices, but if available they provide the doctor a more detailed look at the retina where they can focus and zoom on areas of interest and they can compare images year to year.

---
## [Image Analysis](./code/01-eda.ipynb)

All data for this project is from Kaggle's [Ocular Disease Recognition Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k). This dataset included right and left eye images for 5,000 patients, as well as these images preprocessed where they were standardized to one image size and images of low quality with no diagnosis were removed. I worked with these preprocessed images. Also included is a .csv with the details shown below for each image.

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
<br>

I prepared the data for both a binary classification for health vs. non-healthy retinal images as well as multiclass classification for each individual disease label. Seen below, the distribution for the binary classes was fairly even compared to very imbalanced classes for the multiclass problem.

<center><img src='./images/binary_class_distribution.png' width='500'></center>
<center><img src='./images/multiclass_histogram.png' width='700'></center>
<br>

For most of the diseases, patients were split evenly on whether the condition affected both eyes or only one eye. The exceptions to this were AMD and hypertension where the majority of patients showed signs in both eyes. Those patients that have some other condition or abnormality that doesn't fall under the 6 listed conditions predominantly only have it in one eye.

<center><img src='./images/conditions_between_eyes.png' width='800'></center>
<br>


---
## [Modeling](./code/02_modeling.ipynb)



---
## Insights and Next Steps

