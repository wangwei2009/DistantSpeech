from setuptools import setup, find_packages
import string

# Load description
def find_long_description():
    with open('README.md', 'r') as f:
        return f.read()


# Read the version string
def find_version():
    version_var_name = '__version__'
    with open('DistantSpeech/_version.py', 'r') as f:
        for l in f:
            if not l.startswith(version_var_name):
                continue
            return l[len(version_var_name) :].strip(string.whitespace + '\'"=')
        raise RuntimeError('Unable to read version string.')


setup(
    name='DistantSpeech',
    # version=find_version(),
    version='0.1.0',
    description='A python package for distant speech enhancemnet.',
    author='Wei Wang',
    packages=[
        "DistantSpeech",
    ],
    # packages=find_packages(exclude=('docs',)),
    python_requires='>=3.5',
    install_requires=['numpy>=1.14.0', 'scipy>=1.1.0', 'matplotlib>=2.1.0', 'librosa'],
    zip_safe=False,
)
