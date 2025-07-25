from setuptools import setup, find_packages, Command
from setuptools.command.install import install as _install
# import os
# import shutil
# import glob

# class CleanCommand(Command):
#     """Custom clean command to tidy up the project root."""
#     user_options = []

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def run(self):
#         for dirpath in ('build', 'dist', '*.egg-info'):
#             for path in glob.glob(dirpath):
#                 if os.path.isdir(path):
#                     shutil.rmtree(path)
#                 elif os.path.exists(path):
#                     os.remove(path)

# class InstallAndCleanCommand(_install):
#     """Custom install command to run clean after install."""
    
#     def run(self):
#         _install.run(self)
#         self.run_command('clean')


setup(name='flashcurve',
    version='2.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'flashcurve': ['*.fit'],  
        'flashcurve': ['*.tflite'],  
        'flashcurve': ['*.yaml'], 
    },
    exclude_package_data={
        # '': ['fermi_tools.py'],
        '': ['lc_search_example.py'],
        '': ['make_image_example.ipynb'],
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
    url='https://github.com/kristiantcho/flashcurve',
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    cmdclass={
        # 'clean': CleanCommand,
        # 'install': InstallAndCleanCommand,
    },
    )