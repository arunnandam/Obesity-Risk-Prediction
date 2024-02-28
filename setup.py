from setuptools import find_packages, setup
from typing import List

#define Constant to handle e .
HYPEN_DOT = "-e ."

def get_requirements(file_name:str)->List[str]:
    '''
    This Function will return the list of packages as List
    '''
    requirements = []
    with open(file_name) as file:
        requirements = file.readlines()
        requirements = [pkg.replace("\n","") for pkg in requirements]

    #Handle e.
    if HYPEN_DOT in requirements:
        requirements.remove(HYPEN_DOT)
    
    return requirements


    
setup(
    name="Obesity-Risk-Prediction",
    version='0.0.1',
    author='Arun Nandam',
    author_email='arunkumar.nandam@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)