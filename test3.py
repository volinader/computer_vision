# import the necessary packages
import pandas as pd
import numpy as np
import pytesseract
import argparse
import imutils
import cv2
from imutils.contours import sort_contours
#from yargy.tokenizer import MorphTokenizer
from yargy import rule, Parser, or_, and_, not_
from yargy.predicates import gram
#from yargy.interpretation import fact
#from yargy.relations import gnc_relation

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(H, W) = gray.shape
cv2.imshow("gray", gray)
cv2.waitKey(0)

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 15))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
# smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morpholigical operator to find dark regions on a light
# background
gray = cv2.GaussianBlur(gray, (1, 1), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("Blackhat", blackhat)
cv2.waitKey(0)



# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad_horizontal = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad_horizontal = np.absolute(grad_horizontal)
cv2.imshow("Gradient1", grad_horizontal)
cv2.waitKey(0)
(minVal, maxVal) = (np.min(grad_horizontal), np.max(grad_horizontal))
grad_horizontal = (grad_horizontal - minVal) / (maxVal - minVal)
grad_horizontal = (grad_horizontal * 255).astype("uint8")
cv2.imshow("Gradient", grad_horizontal)
cv2.waitKey(0)

grad_vertical = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
grad_vertical = np.absolute(grad_vertical)
cv2.imshow("Gradient1", grad_vertical)
cv2.waitKey(0)
(minVal, maxVal) = (np.min(grad_vertical), np.max(grad_vertical))
grad_vertical = (grad_vertical - minVal) / (maxVal - minVal)
grad_vertical = (grad_vertical * 255).astype("uint8")
cv2.imshow("Gradient", grad_vertical)
cv2.waitKey(0)
grad = cv2.addWeighted(grad_horizontal, 0.5, grad_vertical, 0.5, 0)
cv2.imshow('Sobel Image',grad)
cv2.waitKey(0)


# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Rect Close", thresh)
# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# series of erosions to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=2)
cv2.imshow("Square Close", thresh)
cv2.waitKey(0)


# find contours in the thresholded image and sort them from bottom
# to top (since the MRZ will always be at the bottom of the passport)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="bottom-to-top")[0]
# initialize the bounding box associated with the MRZ
mrzBox = None
# loop over the contours
union_table = []
union_table_test = []

for c in cnts:
    counter = 0

    # compute the bounding box of the contour and then derive the
    # how much of the image the bounding box occupies in terms of
    # both width and height
    (x, y, w, h) = cv2.boundingRect(c)
    percentWidth = w / float(W)
    percentHeight = h / float(H)
    #print(percentHeight, percentWidth)
    # if the bounding box occupies > 80% width and > 4% height of the
    # image, then assume we have found the MRZ
    if x != 0 and y != 0 and w != 0 and h != 0 and percentWidth < 0.70:
        mrzBox = (x, y, w, h)
        (x, y, w, h) = mrzBox
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX * 2), h + (pY * 2))
        # extract the padded MRZ from the image

        mrz = image[y:y + h, x:x + w]
        cv2.imshow("mrz", mrz)
        cv2.waitKey(0)
        custom_oem_psm_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'

        mrzText = pytesseract.image_to_data(mrz, lang='rus', config=custom_oem_psm_config, output_type=pytesseract.Output.DATAFRAME)
        #print(mrzText)
        mrzText_to_string = pytesseract.image_to_string(mrz, lang='rus', config=custom_oem_psm_config)
        mrzText_to_string = mrzText_to_string.replace(" ", "").replace("\n", "")
        if len(mrzText_to_string) > 4 and len(mrzText_to_string) < 20:

            union_table_test.append(mrzText_to_string)
        union_table.append(mrzText)
        #mrzText = mrzText.replace(" ", "")
        #img_conf_text = mrzText[["conf", "text"]]
        #img_valid = img_conf_text[img_conf_text["text"].notnull()]
        #img_words = img_valid[img_valid["text"].str.len() > 1]

        #print(mrzText)		
        #virtical_img = cv2.rotate(mrz, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #mrzText_vert = pytesseract.image_to_string(virtical_img, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
        #mrzText_vert = mrzText_vert.replace(" ", "")
        #print(mrzText)
print(union_table_test)

table = pd.concat(union_table)
table = table[(table['text'].notna()) & (table['text'].str.len() > 2) & (table['text'].str.len() < 17)]
table = table['text'].tolist()
print(table)
companies = []

counter = 0
for i in table:
        #print(i)
        if i.lower() in ('фамилия', 'имя', 'отчество'):
                companies.append(table[counter+1])
        counter += 1  

COMPANY = rule(or_
                (and_
                (gram('Name'), 
                not_
                (gram('Abbr')),
                not_
                (gram('ADJS'))), 
                and_
                (gram('Patr'), 
                not_
                (gram('Fixd')), 
                not_
                (gram('Abbr'))), 
                and_(gram('Surn'),
                not_
                (gram('Pltm')),
                not_
                (gram('Abbr')))))
                
company_finder = rule(COMPANY)
table = table + union_table_test
parser_company = Parser(company_finder)
for i in table:

    matches = (list(parser_company.findall(i)))
    for i in matches:
        if [k.value for k in i.tokens][0] not in companies:
            companies.append([k.value for k in i.tokens][0])


companies = [x.upper() for x in companies]
print(companies)
'''# if the MRZ was not found, exit the script
if mrzBox is None:
    print("[INFO] MRZ could not be found")
    sys.exit(0)
# pad the bounding box since we applied erosions and now need to
# re-grow it
(x, y, w, h) = mrzBox
pX = int((x + w) * 0.03)
pY = int((y + h) * 0.03)

(x, y) = (x - pX, y - pY)
(w, h) = (w + (pX * 2), h + (pY * 2))
# extract the padded MRZ from the image
mrz = image[y:y + h, x:x + w]
print(mrz)'''


# OCR the MRZ region of interest using Tesseract, removing any
# occurrences of spaces
'''mrzText = pytesseract.image_to_string(mrz, lang='rus')
mrzText = mrzText.replace(" ", "")
print(mrzText)
# show the MRZ image
cv2.imshow("MRZ", mrz)
cv2.waitKey(0)'''

'''# threshold the image using Otsu's thresholding method
thresh = cv2.threshold(blackhat, 50, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Otsu", thresh)
cv2.waitKey(0)



# apply a distance transform which calculates the distance to the
# closest zero pixel for each pixel in the input image
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
# normalize the distance transform such that the distances lie in
# the range [0, 1] and then convert the distance transform back to
# an unsigned 8-bit integer in the range [0, 255]
dist = cv2.normalize(dist, dist, 0, 144.0, cv2.NORM_MINMAX)
dist = (dist * 255).astype("uint8")
cv2.imshow("Dist", dist)
# threshold the distance transform using Otsu's method
dist = cv2.threshold(dist, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Dist Otsu", dist)
cv2.waitKey(0)'''

