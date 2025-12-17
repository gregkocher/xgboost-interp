# Contributing to xgboost-interp

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. **Fork the repository** on GitHub to your own account
2. **Clone your fork** locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/xgboost-interp.git
   cd xgboost-interp
   ```

## Development Setup

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already
2. Install dependencies:
   ```bash
   uv sync
   ```

## Making Changes

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test locally by running the examples:
   ```bash
   uv run python xgboost_interp/examples/iris_classification_example.py
   uv run python xgboost_interp/examples/california_housing_example.py
   ```

3. **Run the tests** to make sure everything works:
   ```bash
   uv run pytest tests/
   ```

4. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

## Submitting a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub against the `main` branch

3. **Fill out the PR template** with:
   - A description of what your PR does
   - The type of change (bug fix, new feature, etc.)
   - Any related issues
   - Confirmation that you've completed the checklist items

## Code Style

- Follow existing code patterns and conventions in the repository
- Use clear, descriptive variable and function names
- Add docstrings to new functions and classes
- Keep changes focused and minimal

## Questions?

If you have questions or need help, feel free to open an issue.

