import torch
from transformers import BertTokenizer, BertForMaskedLM
import re
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

from enchant.checker import SpellChecker
from difflib import SequenceMatcher


def typo_detection_correction(text):
   
    rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', 
            '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ', 
            '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', 
            '(': ' ( ', ')': ' ) ', "s'": "s '"}
   
    text_original = str(text)

    #Cleaning the text recieved from ocr 
    text = re.sub("[,.:!\"I]","",text)
    text_original = re.sub("[,.:!\"I]","",text_original)
    
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    
    def get_personslist(text):
        personslist=[]
        for sent in nltk.sent_tokenize(text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                    personslist.insert(0, (chunk.leaves()[0][0]))
        return list(set(personslist))
    personslist = get_personslist(text)
    ignorewords = personslist + ["!", ",", "\"", "?", '(', ')', '*', "'"]

    d = SpellChecker("en_US")
    words = text.split()
    incorrectwords = [w for w in words if not d.check(w) and w not in ignorewords]

    #using enchant.checker.SpellChecker, get suggested replacements
    suggestedwords = [d.suggest(w) for w in incorrectwords]
    
    # replace incorrect words with [MASK]
    for w in incorrectwords:
        text = text.replace(w, '[MASK]')
        text_original = text_original.replace(w, '[MASK]')
        
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = [t for t in tokenized_text if t!="."]
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']

    # Create the segments tensors
    segs = [i for i, e in enumerate(tokenized_text) if e == "."]
    segments_ids=[]
    prev=-1
    for k, s in enumerate(segs):
        segments_ids = segments_ids + [k] * (s-prev)
        prev=s
    segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
    segments_tensors = torch.tensor([segments_ids])

    # prepare Torch inputs 
    tokens_tensor = torch.tensor([indexed_tokens])

    # Load pre-trained model
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Predict all tokens
    with torch.no_grad():
        output = model(tokens_tensor, segments_tensors)
        predictions = output[0]
  
    def predict_word(text_original, predictions, maskids):
        print("\n",text_original)
        pred_words=[]
        for i in range(len(MASKIDS)):
            probs = torch.nn.functional.softmax(predictions[0, MASKIDS[i]], dim=-1)
            preds = torch.topk(probs, k=50) 
            indices = preds.indices.tolist()
            list1 = tokenizer.convert_ids_to_tokens(indices)
            list2 = suggestedwords[i]
            simmax=0
            predicted_token='' 
            for word1 in list1:
                for word2 in list2:
                    s = SequenceMatcher(None, word1, word2).ratio()
                    if s is not None and s > simmax:
                        simmax = s
                        predicted_token = word1
            if(len(predicted_token)==0):
                     predicted_token = list1[0]
                     if(len(list2)==0):
                         predicted_token = list2[0]
                          
            text_original = text_original.replace('[MASK]', predicted_token, 1)
        return text_original
    
    text_original = predict_word(text_original, predictions, MASKIDS)
    return text_original   