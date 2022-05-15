from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

version = {}
with open("./hahtorch/_version.py") as fp:
    exec(fp.read(), version)

release_tag = "v_" + str(version['version_info'][0]) + \
    str(version['version_info'][1]) + str(version['version_info'][2])

setup(
    name='hahtorch',
    packages=['hahtorch'],
    version=version['__version__'],
    license='MIT',
    description='Hebbian/Anti-Hebbian Learning for Pytorch',
    long_description_content_type="text/markdown",
    long_description=long_description,
    author='Metehan Cekic',
    author_email='metehancekic@ucsb.edu',
    url='https://github.com/metehancekic/HaH.git',
    download_url='https://github.com/metehancekic/HaH/archive/' + release_tag + '.tar.gz',
    keywords=['Hebbian', 'Anti-Hebbian', 'Pytorch', 'Neuro-inspired'],
    install_requires=[
        'numpy',
        ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ],
)
