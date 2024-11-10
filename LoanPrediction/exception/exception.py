import sys
import logging
import traceback


# Function to extract detailed error message and traceback
def error_message_details(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        # Detailed string with the traceback details
        return (
            f"Error occurred in script {file_name}, "
            f"line number {line_number}, "
            f"error message: {str(error)}, "
            f"traceback: {''.join(traceback.format_exception(exc_type, exc_value, exc_tb))}"
        )
    else:
        return f"Error occurred: {str(error)}"  # Fallback if no traceback available


# Custom exception class for LoanException
class LoanException(Exception):
    def __init__(self, error_message, error_details=None):
        if error_details is None:
            error_details = sys.exc_info()  # Only get the traceback if necessary
        super().__init__(error_message)
        # Get detailed error message from the error_detail
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message
