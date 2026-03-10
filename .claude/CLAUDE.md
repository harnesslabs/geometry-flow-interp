## Code Environment

- See @README.md for setup and run commands. **Always** use `uv run python example.py` or `uv run python -m ...` for python.
- Use astral (uv, ty, ruff) for best practices. After modifying any Python file, **always** run:
  ```bash
  uv run ruff format <file> # format the file
  uv run ruff check --fix # fix linting issues
  uv run ty check # check for type errors
  ```
- Use PyTorch for training. Prioritize accelerators in this order: cuda, mps, or cpu.

## Philosophy

- Embrace simplicity and elegance in design.
- Culminate ideas from machine learning, mathematics, physics, creativity/curiosity, and human intuition.
- Explore unique and novel approaches to solving complex problems that may be unconventional.
