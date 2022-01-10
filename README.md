# Optical Character Recognition with NLP

An Optical Character Reader for extracting text from images and scanned handwritten text.<br/>

❖ Text from Images Using Tesseract <br/>
❖ Text from handwritten Images Using TensorFlow <br/>

Natural Language Processing (NLP) techniques used to improve OCR accuracy.<br/>

❖ Using BERT(Bidirectional Encoder Representations from Transformers)<br/>
❖ Using NLTK<br/>
❖ Using Python Spellchecker

## OCR Using Tesseract
❑ Tesseract is used directly using an API to extract printed text from images.

❑ Tesseract includes a new neural network subsystem and uses LSTM.

❑ Doesn’t work well while extracting handwritten text.

## OCR Using TensorFlow <br/>
➢ OCR for extracting text from images containing handwritten text. <br/>

➢ Consists of a Neural Network (NN) which is trained using images containing handwritten text from the IAM dataset.<br/>

➢ Image is split line-wise for text extraction, as the model is trained for extracting text from a line.<br/>

### Model Overview
Model consists of : <br/>

--> Convolutional NN (CNN) layers <br/>

--> Recurrent NN (RNN) layers <br/>

--> Connectionist Temporal Classification (CTC). <br/>

## Post-OCR Error Detection and Correction

**I. Process scanned image using OCR**

  ✓ Scanned text is cleaned by removing special and unwanted characters using NLTK library functions.
             
**II. Process document and identify unreadable words**

   ✓ Incorrect words are identified by Python enchant’s SpellChecker function.<br/>
   ✓ NLTK’s “Parts of Speech” tagging is used to exclude person names from incorrect words.<br/>
   ✓ Each incorrect word is replaced with a [MASK] token, and replacement word suggestions from SpellChecker are stored.<br/>

            
**III. Load BERT model and predict replacement words**
          
   ✓ BERT model looks for the [MASK] tokens and then predicts the original value of the masked words, based on the context provided by the other words in the sequence.

**IV. Refine BERT predictions by using suggestions from Python SpellChecker**
            
   ✓ The suggested word list from SpellChecker, which incorporates characters from the garbled OCR output, is combined with BERT’s context-based suggestions to yield                better predictions and the best prediction replaces the [MASK] token.

## Sample Output
![Sample 1](https://user-images.githubusercontent.com/78135321/148759040-50cadf66-6ded-48ba-8d1b-f0d475ae507d.png)

   
   
