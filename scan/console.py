from rich.console import Console

# markup=False / highlight=False: everything printed here is either untrusted model output or an
# exception message derived from it. With Rich's defaults a span like "[high]" is parsed as a style
# tag and silently deleted, and an unmatched "[/]" raises MarkupError -- including from inside the
# error handlers, which re-print the offending text and so let the exception escape them.
# soft_wrap=True stops Rich inserting real newlines at the terminal width, which was mangling the
# markdown report whenever it was redirected to a file.
console = Console(markup=False, highlight=False, soft_wrap=True)
