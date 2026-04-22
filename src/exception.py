import sys
#You are creating your own type of error instead of using built-in ones like ValueError.
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]"

    return error_message.format(file_name, exc_tb.tb_lineno, str(error))


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    





#simpler version ofwhat we did == INTERVIEW oritented
    # class CustomException(Exception):
    #     def __init__(self, message):
    #     super().__init__(message)

    # def __str__(self):
    #     return f"Custom Error: {self.args[0]}"

# def divide(a, b):
#     if b == 0:
#         raise CustomException("Cannot divide by zero")
#     return a / b