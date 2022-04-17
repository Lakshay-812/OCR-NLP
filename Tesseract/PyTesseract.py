import argparse
from path import Path
from PIL import Image
import pytesseract
from pytesseract import image_to_string
from text_detection import east
from Bert import typo_correction
import sys


def main(filename,mode,detector):
#def main():
    
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image', help='Image to read', type=Path,default = "./images/para.png")
    # parser.add_argument('--mode', choices=['tesseract', 'bert'], default='tesseract')
    # parser.add_argument('--detector', choices=['east'],default="None")
    # args = parser.parse_args()
    # filename = args.image
    # mode = args.mode
    # detector = args.detector
    
    text_original = ""

    '''Extracts text when the text is spread in the background''' 
    if(detector == "east"):
        n = east(filename)
        for i in range(n+1):
            text = image_to_string(Image.open("result{}.png".format(i))) 
            text_original+= text+ "\n"
       
    else: 
        '''Extracts Text from the Image'''          
        text = image_to_string(Image.open(filename))    
        text_original = str(text)
    

    '''Bert used for improving OCR accuracy'''  
    if(mode == "bert"):

        print(text_original)
        a = "\n"+typo_correction(text,text_original)
        
        print(a)
        return(a)
    
    else:
        '''Text extracted by Tesseract'''    
        print(text_original)
        return ("\n",text_original)    


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    #main()
  
    


