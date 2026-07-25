"""The two output streams SCAN writes to.

Everything that is not the report goes to stderr, so ``run-scan > report.md`` yields a clean
document. The report itself is written to stdout by :func:`scan.report.emit`.
"""

from rich.console import Console

# markup=False / highlight=False: this console prints exception messages derived from model
# output. With Rich's defaults a span like "[high]" is parsed as a style tag and silently
# deleted, and an unmatched "[/]" raises MarkupError -- including from inside the error handler
# that is trying to report it, which is how the exception escaped both handlers.
# soft_wrap=True stops Rich inserting real newlines at the terminal width.
console = Console(stderr=True, markup=False, highlight=False, soft_wrap=True)
