import torch
from transformers import BertTokenizer, BertForMaskedLM
import re
import nltk
# nltk.download('punkt')
from enchant.checker import SpellChecker
from difflib import SequenceMatcher
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


def typo_correction(text,text_original):
    
    '''Data Prepocessing and cleanup'''
    
    rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', 
            '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ', 
            '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', 
            '(': ' ( ', ')': ' ) ', "s'": "s '"}

    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    
    '''Using nltk's parts-of-speech tagging(chunking) to exclude person names'''
    def get_personslist(text):
        personslist=[]
        for sent in nltk.sent_tokenize(text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                    personslist.insert(0, (chunk.leaves()[0][0]))
        return list(set(personslist))
    
    personslist = get_personslist(text)
    ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'"]

    '''Use Spellchecker to identify incorrect words'''
    d = SpellChecker("en_US")
    words = text.split()
    incorrectwords = [w for w in words if not d.check(w) and w not in ignorewords]


    '''Using Enchant SpellChecker to get suggested replacements'''
    suggestedwords = [d.suggest(w) for w in incorrectwords]


    ''' Replacing incorrect words with [MASK]'''
    for w in incorrectwords:
        text = text.replace(w, '[MASK]')
        text_original = text_original.replace(w, '[MASK]')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   
    '''Tokenizing our text'''
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = [t for t in tokenized_text if t!="."]

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']


    '''Creating the segments tensors to distinguish multiple sentences'''
    segs = [i for i, e in enumerate(tokenized_text) if e == "."]
    segments_ids=[]
    prev=-1
    for k, s in enumerate(segs):
        segments_ids = segments_ids + [k] * (s-prev)
        prev=s
    segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
    segments_tensors = torch.tensor([segments_ids])

    '''Torch inputs''' 
    tokens_tensor = torch.tensor([indexed_tokens])
   
        
    print("In Bert")
    '''Load pre-trained model'''
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    print("Out Bert")


    ''' Predicting all tokens'''
    with torch.no_grad():
        output = model(tokens_tensor, segments_tensors)
        predictions = output[0]
    
    text_original = predict_word(text_original, predictions, MASKIDS,tokenizer,suggestedwords)

    return text_original    

'''Combining BERT's suggestions with SpellChecker's suggestions'''
def predict_word(text_original, predictions, MASKIDS,tokenizer,suggestedwords):
        print(text_original)
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
            text_original = text_original.replace('[MASK]', predicted_token, 1)
        return text_original    