from setuptools import setup, find_packages

install_requires = open("requirements.txt", 'r').read().split('\n')
install_requires = [str(ir) for ir in install_requires]

setup(name='pytorch_bio_transformations', version='0.0.2',
    description='Pytorch Biologically Motivated Transformations>', url='https://CeadeS.github.io/pytorch_bio_transformations"',
    author='Martin Hofmann', author_email='Martin.Hofmann@tu-ilmenau.de', license='MIT License',
    packages=find_packages(), install_requires=install_requires, python_requires=">=3",
    keywords="python, biomodule, pytorch",

    classifiers=['Development Status :: 1 - Planning', 'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', 'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7', 'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10', 'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12', ], )
