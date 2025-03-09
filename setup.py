from setuptools import setup, find_packages

install_requires = open("requirements.txt", 'r').read().split('\n')
install_requires = [str(ir) for ir in install_requires]

setup(
    name='pytorch_bio_transformations',
    version='0.0.5',
    description='PyTorch Biologically Motivated Transformations',  # Fixed typo (removed '>')
    url='https://CeadeS.github.io/pytorch_bio_transformations',  # Fixed URL (removed extra quote)
    author='Martin Hofmann',
    author_email='Martin.Hofmann@tu-ilmenau.de',
    license='MIT License',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",  # Made Python requirement more specific
    keywords="python, biomodule, pytorch, neural networks, biologically-inspired",  # Added relevant keywords

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)