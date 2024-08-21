"""This module implements our CI function calls."""

import nox


@nox.session(name="test")
def run_test(session):
    """Run pytest."""
    session.install("-r", "requirements.txt")
    session.run("pytest")
    session.install("pytest")


@nox.session(name="testgit")
def run_test_git(session):
    """Run pytest."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "./tests/test_brain_decoder.py", "./tests/test_function.py")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8==4.0.1")
    session.install(
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.install("bandit==1.7.2")
    session.run("flake8", "src/train_brain_decoder.py", "tests", "noxfile.py")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install("-r", "requirements.txt")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--no-strict-optional",
        "--no-warn-return-any",
        "--implicit-reexport",
        "--allow-untyped-calls",
        "src/train_brain_decoder.py",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "tests", "noxfile.py")
    session.run("black", "src", "tests", "noxfile.py")
