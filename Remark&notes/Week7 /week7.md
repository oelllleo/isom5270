# Week7 

Fundamental Concept 

- NNN's representation, evaluation and optimization
- Deep learing on Text

## NNN's representation, evaluation and optimization

## Deep learning on Text

### Word Embedding

Classify the words to related things 
e.g. prince and Princese  all related to same category 

Embedding =  mapping the relationship of 2 meaning 

e.g. word to idea. song to playlist, pixels to photos

#### Why are they useful 

the key is hidden layer 

It input a words, and predict the words nearby (+-4)

Details: https://medium.com/@zafaralibagh6/a-simple-word2vec-tutorial-61e64e38a6a1

### Encoder and decoder

Image -> Image Encoder -> image embeding -> Image Decoder -> image

Text->Text Encoder -> text embeding

if image embeding is similair to text embeding
we could make a good use on it 

Text->text Encoder -> text embeding -> Image Decoder-Image

so that text could be image 

could be on other way 

Image -> Image Encoder -> image embeding -> Image Decoder-Image

### LLM

ChatGPT is LLM 

A sequence of works(the quick brown) -> LLM -> next word(Fox)

Feature: embedding 

