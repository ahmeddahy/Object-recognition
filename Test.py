from xlwt import Workbook
import xlsxwriter
from xlrd import open_workbook
from RBF import *

file = open_workbook('ObjectRecognition.xls')
testing_sheet = file.sheet_by_name("Testing")
training_sheet = file.sheet_by_name("Training")
x = rbf(11, training_sheet, testing_sheet)
x.initial_values(.001,.9,16)
x.train()
print(x.centers)

