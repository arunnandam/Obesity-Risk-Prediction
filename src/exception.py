# To handle all the exceptions.

import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    #return type of sys
    _,_,exc_tb = error_detail.exc_info()

    #Getting file name where exception is occured 
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Defining Error Message
    error_message = "Error Occured in Python Script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        # Since we are building custom exception, we need to inherit from the exception.
        super().__init__(error_message)
        # The error is formatted and stored back in variable.
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

# Test Code to test Exception.py  
if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero Error")
        raise CustomException(e,sys)
