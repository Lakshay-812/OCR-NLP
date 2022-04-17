import cv2


def split(img_path):
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rotated=gray 

    ## (2) threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    # (3) minAreaRect on the nozeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx,cy), (w,h), ang = ret
    if w<h:
        w,h = h,w
        ang -= 90

    # (4) Find rotated matrix, do rotation

    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))


    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG)

    th = 2
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
  
    i = 0
    while(i<len(uppers)-1):
        if(uppers[i+1]-lowers[i]<=5):
            uppers.remove(uppers[i+1])
            lowers.remove(lowers[i])
        else:
            i+=1    
    
    count = -1
    if(len(lowers)==0): 
        lowers.append(H-10)

    if(len(uppers)==0): 
        uppers.append(10)    
    
    if(len(uppers)==1): 
            uppers[0]+=10
    
    if(len(lowers)==1): 
            lowers[0]-=10        

    
    for i in range(len(uppers)):
        if(-uppers[i]+20+lowers[i]>=20):
            count+=1
            if(uppers[i]>10 and lowers[i]+10<=H):
                cv2.imwrite(r"..\data\temp\result{}.png".format(count), rotated[uppers[i]-10:lowers[i]+10])
            else:
                 cv2.imwrite(r"..\data\temp\result{}.png".format(count), rotated[uppers[i]:lowers[i]])
             
    return count    