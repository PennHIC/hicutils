from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt') as fh:
    install_requires = fh.read().splitlines()


setup(
    name='hicutils',
    version='0.0.3',
    author='Aaron M. Rosenfeld',
    author_email='aaron.rosenfeld@pennmedicine.upenn.edu',
    url='https://github.com/PennHIC/hicutils',
    packages=[
        'hicutils',
        'hicutils.core',
        'hicutils.plots',
    ],
    install_requires=install_requires
)
