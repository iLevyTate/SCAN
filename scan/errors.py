# scan/errors.py


class MissingEnvironmentVariableError(Exception):
    """Exception raised when a required environment variable is missing."""

    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        self.message = f"The environment variable '{variable_name}' is missing."
        super().__init__(self.message)
