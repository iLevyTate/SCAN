import pytest

from scan.console import console


@pytest.mark.parametrize(
    "text",
    (
        "Cost estimate [see appendix], risk [high].",
        "Checklist: [ ] vet  [x] food",
        "Insert [your name] here",
        "Time budget [hours/week]",
    ),
)
def test_bracketed_model_output_is_printed_verbatim(text, capsys):
    # Regression: Rich's default markup=True parsed these spans as style tags and deleted
    # them, silently changing the report the user paid for.
    console.print(text)

    assert capsys.readouterr().out.rstrip("\n") == text


def test_an_unmatched_closing_tag_does_not_raise(capsys):
    # Regression: "[/]" raised MarkupError. Both error handlers re-print the exception
    # message, which embeds the offending tag, so the error escaped them and killed the CLI
    # after the crew had already completed.
    console.print("Checklist: [/] vet visit done.")

    assert "[/]" in capsys.readouterr().out


def test_long_lines_are_not_hard_wrapped(capsys):
    # Regression: Rich re-wrapped at the console width, injecting real newlines into the
    # markdown report whenever run-scan was redirected to a file.
    paragraph = "word " * 60

    console.print(paragraph)

    assert capsys.readouterr().out.count("\n") == 1
