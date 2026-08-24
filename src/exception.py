import sys
import os
from src.logger import logging
from datetime import datetime
from typing import Any
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb is not None else "<unknown>"
    error_message ="Error occured in python script name[{0}] line number [{1}] error messsage [{2}]".format(
     file_name,exc_tb.tb_lineno if exc_tb is not None else "<unknown>",str(error))
     
    return error_message



        
    
class CustomException(Exception):
    def __init__(self,error_meesage, error_details:sys):
        super().__init__(error_meesage)
        self.error_message=error_message_detail(error_meesage,error_detail=error_details)
        
        
    def __str__(self):
        return self.error_message
    
    
    
# if __name__ == "__main__":
    
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("divide by zero")
#         raise CustomException(e,sys)