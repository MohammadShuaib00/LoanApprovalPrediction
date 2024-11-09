import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def error_message_details(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    # Return a string with all the error details
    return (
        f"Error occurred in script {file_name}, "
        f"line number {line_number}, "
        f"error message: {str(error)}"
    )


class LoanException(Exception):
    def __init__(self, error_message, error_details=sys.exc_info()):
        super().__init__(error_message)
        # Get the detailed error message
        self.error_message = error_message_details(error_message, error_details)

        # Log the error message
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message


# Example usage
try:
    raise ValueError("This is a custom error for testing!")
except ValueError as e:
    raise LoanException(str(e), sys.exc_info())
