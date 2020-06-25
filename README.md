# bert_keras_multiclass_5labels
A simple multiclass text classification model I have built from scratch using tensorflow/keras, and using transfer learning to make use of Google BERT. 

The model trains well with a prediction accuracy of 87%. 

I found that removing some specific words (via a quick wordcloud for each label) helped increase the accuracy of the model. Words were chosen for their non-specific nature yet possibly specific to the context of a couple of labels, hence causing some confusion. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dd5edf11-7c00-44d7-a6e3-1c2856593df6/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dd5edf11-7c00-44d7-a6e3-1c2856593df6/Untitled.png)
