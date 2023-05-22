from setuptools import setup

setup(
	name='cma',
	version='0.0.1',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'SVM_prediction=cma.SVM_prediction:main'
        ]
    },
    install_requires=['h5py','numpy','pandas','scikit-learn','scanpy','rpy2','importlib-resources'],
    package_data={"cma.data": ["*.csv", "*.h5ad"]},
)
