## Code Environment

- Use astral (uv, ty, ruff) for best practices. Do not commit changes unless formatting, linting, and type checks pass.
  ```bash
  # install dependencies
  uv sync
  # add dependencies
  uv add <package>
  # run scripts
  uv run python -m gf.main             # training
  uv run python -m gf.generate         # generation
  # after modifying any Python file, **always** run
  uv run ruff format <file> # format the file
  uv run ruff check --fix # fix linting issues
  uv run ty check # check for type errors
  ```
- Use PyTorch for training. Prioritize accelerators in this order: cuda, mps, or cpu.

## Philosophy

- Embrace simplicity and elegance in design.
- Culminate ideas from machine learning, mathematics, physics, creativity/curiosity, and human intuition.
- Explore unique and novel approaches to solving complex problems that may be unconventional.
