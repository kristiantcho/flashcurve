from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='flashcurve',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'flashcurve': ['*.fits'],  
        'flashcurve': ['*.tflite'],  
        'flashcurve': ['*.yaml'], 
    },
    exclude_package_data={
        # '': ['fermi_tools.py'],
        '': ['lc_search.py'],
    },
    install_requires=['numpy',
                'pandas',
                'astropy',
                'matplotlib',
                'mechanize',
                'requests',
                ],
    entry_points={
        'console_scripts': [
            # Define script entry points if needed
            # 'script_name=module:function',
        ],
    },
    author='Kristian Tchiorniy',
    author_email='kristiantcho@gmail.com',
    description='flashcurve: A machine-learning approach for the fast generation of adaptive-binning lightcurves with Fermi-LAT data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_project',
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',)