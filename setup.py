from setuptools import setup,find_packages
from typing import List

Minus_E_Dot = "-e ."

def get_requirements(file_path:str)->list[str]:
    '''Will import all the library from here'''
    
    requirements =[]
    
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.strip() for req in requirements]
        
    if Minus_E_Dot in requirements:
        requirements.remove(Minus_E_Dot)
        
    return requirements


setup(
    name = 'Realtime_fraud_detection',
    version = '0.0.1',
    author='Ramands7',
    author_email='dsraman07@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_required=">=3.11"
    
    
)
        
    