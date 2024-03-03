import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('1.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

hImg, wImg,_ = img.shape
boxes = pytesseract.image_to_data(img)
print(boxes)
for x, b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        print(b)
        if len(b)==12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x, y), (w+x, h+y), (0, 0, 255), 1)
            cv2.putText(img, b[11], (x, y), cv2.FONT_ITALIC, 0.75, (139, 0, 0), 2)

cv2.imshow('Text detection', img)
cv2.waitKey(0)
