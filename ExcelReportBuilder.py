from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import ColorScaleRule

import io

import pandas as pd

class ExcelReportBuilder:


    def __init__(self, file_name): 
        self.images_ = []
        self.tables_ = []
        self.file_name_ = file_name 

    def AddImage(self, stream: io.BytesIO, page_name: str, position: str="A1"):
        self.images_.append({'page': page_name, 'image': Image(stream), 'position': position})
    
    def AddTable(self, table: pd.DataFrame, page_name: str, drop_index=False, conditional_formatting=False):
        self.tables_.append({'page': page_name, 'table': table, 'index': not(drop_index), 'conditional_formatting': conditional_formatting})

    @staticmethod
    def __GetSheetPtr(wb: Workbook, name: str):
        if name not in wb.sheetnames:
            return wb.create_sheet(name)
        else: 
            return wb[name]

    
    def SaveToFile(self):
        wb = Workbook()
        
        for tab in self.tables_:
            ws = ExcelReportBuilder.__GetSheetPtr(wb, tab['page'])
            start_row = ws.max_row + 1
            for r in dataframe_to_rows(tab['table'], index=tab['index'], header=True):
                ws.append(r)
            if tab['conditional_formatting']:
                formatting_range = '{}{}:{}'.format(
                    ('A' if not tab['index'] else 'B'), 
                    start_row + 1, 
                    ws.cell(ws.max_row, ws.max_column).coordinate
                )
                ws.conditional_formatting.add(formatting_range, 
                    ColorScaleRule(
                        start_type='min', start_value=10, start_color='F8696B',
                        mid_type='percentile', mid_value=50, mid_color='FFEB84',
                        end_type='max', end_value=90, end_color='63BE7B'
                    ))
            ws.append([])

        for img in self.images_:
            ws = ExcelReportBuilder.__GetSheetPtr(wb, img['page'])
            ws.add_image(img['image'], img['position'])

        del wb['Sheet']
        wb.save(self.file_name_)

