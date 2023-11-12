# STREAMER Documentation

This is the source for the [STREAMER Documentation]() written in Python 3.10 with
[Sphinx](https://www.sphinx-doc.org/en/master/).

## Set up

```bash
> conda env create --file environment.yml # root folder
> conda activate streamer
> pip install -r requirements.txt # required for building this documentation
```

**Making the documentation**

```bash
> cd docs
> make html
```
