# LaTeX Setup

## Install MacTeX

```bash
# Install MacTeX via Homebrew
brew install --cask mactex

# Run the installer (will open GUI)
open /opt/homebrew/Caskroom/mactex/2025.0308/mactex-*.pkg
```

Follow the installer prompts and enter your password when asked.

## Compile Paper

After installation completes, restart your terminal and test:

```bash
./compile.sh
```

The PDF should be generated successfully.

## Compile Poster

After installation completes, restart your terminal and run:

```bash
./compile_poster.sh
```

The poster PDF will be generated at `poster.pdf`.
