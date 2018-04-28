from xlwt import Workbook
import xlsxwriter
from xlrd import open_workbook
from RBF import *
import xlwt
'''
file = open_workbook('ObjectRecognition.xls')
testing_sheet = file.sheet_by_name("Testing")
training_sheet = file.sheet_by_name("Training")
x = rbf(13, training_sheet, testing_sheet)
x.initial_values(.001, .5, 300)
x.train()
print(x.test())
file = open("DOD.txt", "w")
for i in x.avg_list:
    a = str(i)
    file.write(a)
    file.write(" ")
file.write("\n")
for i in x.mx_list:
    a = str(i)
    file.write(a)
    file.write(" ")
file.write("\n")
for i in x.mn_list:
    a = str(i)
    file.write(a)
    file.write(" ")
file.write("\n")
for i in x.centers:
    for j in i:
        a = str(j)
        file.write(a)
        file.write(" ")
    file.write("\n")
print(len(x.weights))
for i in x.weights:
    for j in i:
        a = str(j)
        file.write(a)
        file.write(" ")
    file.write("\n")
file.close()'''
