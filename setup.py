
import os
from setuptools import setup, find_packages

import multiview_generator

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

def setup_package():
    """Setup function"""

    name = 'multiview_generator'
    version = multiview_generator.__version__
    description = 'A fully customizable multiview dataset multiview_generator'
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as readme:
        long_description = readme.read()
    group = 'dev'
    url = 'https://gitlab.lis-lab.fr/{}/{}'.format(group, name)
    project_urls = {
        'Documentation': 'http://{}.pages.lis-lab.fr/{}'.format(group, name),
        'Source': url,
        'Tracker': '{}/issues'.format(url)}
    author = 'Baptiste Bauvin and Dominique Benielli and Sokol Koco'
    author_email = 'baptiste.bauvin@lis-lab.fr'
    license = 'GNUGPL'
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved",
        "Natural Language :: French",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Machine Learning",],
    keywords = ('machine learning, supervised learning, classification, '
                'multiview', 'data', 'generation')
    packages = find_packages(exclude=['*.tests'])
    install_requires = requirements
    python_requires = '>=3.5'
    extras_require = {
        'dev': ['pytest', 'pytest-cov'],
        'doc': ['sphinx', 'numpydoc', 'sphinx_gallery', 'matplotlib']}
    include_package_data = True

    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          url=url,
          project_urls=project_urls,
          author=author,
          author_email=author_email,
          license=license,
          classifiers=classifiers,
          keywords=keywords,
          packages=packages,
          install_requires=install_requires,
          python_requires=python_requires,
          extras_require=extras_require,
          include_package_data=include_package_data)


if __name__ == "__main__":
    setup_package()
