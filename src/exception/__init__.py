import sys
import os

def error_messsage_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    error_file = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_line_no = exc_tb.tb_lineno
    return f"Error occured in python script name {error_file}, {error_line_no}, {str(error)}"

class SpamDetectorException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_messsage_detail(error_message, error_detail= error_detail)

    def __str__(self):
        return self.error_message
