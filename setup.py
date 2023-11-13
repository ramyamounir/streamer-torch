from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

setup(
  name="streamer-torch",
  version="0.0.1",
  build="0",
  author="Ramy Mounir",
  url="https://ramymounir.com/docs/streamer/",
  description=r"""Official implementation of STREAMER, a self-supervised hierarchical event segmentation and representation learning""",
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: Free for non-commercial use",
  ],
  python_requires='>=3.10',
  install_requires=[
      'torch>=2.0.0',
      'ddpw>=5.1.0',
      'opencv-python-headless',
      'tqdm',
      'scikit-video',
      'matplotlib',
      'torchvision',
      'tensorboard',
      'pandas',
      'moviepy',
      'librosa',
      'pillow',
      ]
)
