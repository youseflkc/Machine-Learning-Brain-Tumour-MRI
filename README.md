# Brain-Tumour-MRI-AI
Download data set from: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Sample Images:

![image](https://user-images.githubusercontent.com/28910967/209209720-8c992a75-e5b4-4f3d-b1e8-6c903545a6d8.png)

This machine learning model was created using tensorflow. The training model contains 3 convolutional layers. The first convolutional layer contained 16 filters, the
second contained 32 filters, and the final layer contained 64
filters. Between each convolutional layer we placed max-
pooling layers which added some translation invariance to the
training model. The training model was trained on a set of about 4000 images.
![image](https://user-images.githubusercontent.com/28910967/209209558-4a27a2ff-c7d3-43e3-b3df-42f4c8e07186.png).

The model achieved an accuracy of over 95% on 15 epochs and a validation accuracy of over 85%. 

![image](https://user-images.githubusercontent.com/28910967/209209691-414feb01-5457-4c85-b2df-567bd6baeb42.png)

The model was tested on a set of about 400 images and the prediction results are plotted on a confusion matrix and classification report. The model achieved a prediction accuracy of around 70%.

![image](https://user-images.githubusercontent.com/28910967/209209779-10984ff4-5055-48a1-9cd1-6de5e2d847e6.png)
![image](https://user-images.githubusercontent.com/28910967/209209792-fb2a1b56-44ab-4cc1-b60c-834a86de017c.png)
