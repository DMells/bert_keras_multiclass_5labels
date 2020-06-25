# bert_keras_multiclass_5labels
A multiclass text classification model I have built from scratch using tensorflow/keras, and using transfer learning to make use of Google BERT. 

The model trains well with a prediction accuracy of 87% when the data set is reduced to 5 labels.

I found that removing some specific words (via a quick wordcloud for each label) helped increase the accuracy of the model. Words were chosen for their non-specific nature yet possibly specific to the context of a couple of labels, hence causing some confusion. 

Note there are several superfluous features/imports in the code as this is just an extract from a wider sandbox repo. 
