import cv2 as cv
import practices.scripts.operations_over_points as oop

P1 = cv.imread("images/foto1.jpg")
P1 = cv.cvtColor(P1, cv.COLOR_BGR2RGB)
P1 = cv.cvtColor(P1, cv.COLOR_RGB2GRAY)

hpa = oop.miHPA(P1)

print(hpa[0:7])





