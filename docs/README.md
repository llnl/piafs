# PIAFS Documentation

This directory contains the Sphinx-based documentation for PIAFS.

## Building the Documentation

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or on HPC systems, you might need:

```bash
python3 -m pip install --user -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be in `build/html/`. Open `build/html/index.html` in a web browser.

### Build PDF Documentation

```bash
make latexpdf
```

Requires a LaTeX installation. The PDF will be in `build/latex/PIAFS.pdf`.

### Clean Build

```bash
make clean
```

## Viewing the Documentation

After building:

```bash
# Linux/Mac
open build/html/index.html

# Or use a web browser
firefox build/html/index.html
```

## ReadTheDocs

This documentation is configured for ReadTheDocs hosting. See `.readthedocs.yaml` in the project root.

## Documentation Structure

```
docs/
├── source/
│   ├── getting-started/  # Installation and quick start
│   ├── user-guide/       # Comprehensive user documentation
│   ├── examples/         # Example problems and tutorials
│   ├── api/              # API reference
│   ├── developer/        # Developer documentation
│   └── about/            # License, citation, etc.
├── requirements.txt      # Python dependencies
├── Makefile             # Build automation
└── README.md            # This file
```

## Contributing to Documentation

To add or update documentation:

1. Edit `.rst` or `.md` files in `source/`
2. Rebuild: `make html`
3. View changes in your browser
4. Submit a pull request

## Troubleshooting

**Sphinx not found:**
```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

**Build errors:**
- Check that all referenced files exist
- Verify internal links with `:doc:` syntax
- Run `make clean && make html` for a fresh build

## Additional Formats

- **HTML:** `make html` (default)
- **PDF:** `make latexpdf`
- **ePub:** `make epub`
- **Man pages:** `make man`

See `make help` for all available formats.
