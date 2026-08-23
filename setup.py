from setuptools import setup,find_packages
from typing import List

def get_rquirements(file_path:str)->list(str):
    '''Will import all the library from here'''
    
    requirements =[]
    
    with open (file_path) as file_obj:
        requirements=file_obj.readline()
        requirements=[req.strip() for req in requirements]
        
    return requirements


setup(
    name = 'Realtime_frad_detection',
    version = '0.0.1',
    author='Ramands7',
    author_email='dsraman07@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
    
)
        
     