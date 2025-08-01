# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2020
# -----------------
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2020 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
#
# Description:
# -----------
#
#
#
#
#
# Version:
# -------
#
# * multiview_generator version = 0.0.dev0
#
# Licence:
# -------
#
# License: New BSD License
#
#
# ######### COPYRIGHT #########
import os, re
import shutil
from setuptools import setup, find_packages
from distutils.command.clean import clean as _clean
from distutils.dir_util import remove_tree
from distutils.command.sdist import sdist
import multiview_generator

try:
    import numpy
except:
    raise 'Cannot build iw without numpy'
    sys.exit()

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything supprimer de la liste '^core\.*$',
relist = ['^.*~$', '^#.*#$', '^.*\.aux$', '^.*\.pyc$', '^.*\.o$']
reclean = []
USE_COPYRIGHT = True
try:
    from copyright import writeStamp, eraseStamp
except ImportError:
    USE_COPYRIGHT = False

###################
# Get Multimodal version
####################
def get_version():
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    return v_dict["multiview_generator"]

########################
# Set Multimodal __version__
########################
def set_version(multiview_generator_dir, version):
    filename = os.path.join(multiview_generator_dir, '__init__.py')
    buf = ""
    for line in open(filename, "rb"):
        if not line.decode("utf8").startswith("__version__ ="):
            buf += line.decode("utf8")
    f = open(filename, "wb")
    f.write(buf.encode("utf8"))
    f.write(('__version__ = "%s"\n' % version).encode("utf8"))

for restring in relist:
    reclean.append(re.compile(restring))


def wselect(args, dirname, names):
    for n in names:
        for rev in reclean:
            if (rev.match(n)):
                os.remove("%s/%s" %(dirname, n))
        break


######################
# Custom clean command
######################
class clean(_clean):
    def walkAndClean(self):
        os.walk("..", wselect, [])
        pass

    def run(self):
        clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('iw'):
            for filename in filenames:
                if (filename.endswith('.so') or
                        filename.endswith('.pyd') or
                        filename.endswith('.dll') or
                        filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


##############################
# Custom sdist command
##############################
class m_sdist(sdist):
    """ Build source package

    WARNING : The stamping must be done on an default utf8 machine !
    """

    def run(self):
        if USE_COPYRIGHT:
            writeStamp()
            sdist.run(self)
            # eraseStamp()
        else:
            sdist.run(self)

def setup_package():
    """Setup function"""

    name = 'multiview_generator'
    version = get_version()
    multiview_generator_dir = 'multiview_generator'
    set_version(multiview_generator_dir, version)
    description = 'MAGE : Multi-view Artificial Generation Engine, a non-naïve ' \
                  'multiview dataset generator '
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.rst'), encoding='utf-8') as readme:
        long_description = readme.read()
    group = 'dev'
    url = 'https://gitlab.lis-lab.fr/{}/{}'.format(group, name)
    project_urls = {
        'Documentation': 'http://{}.pages.lis-lab.fr/{}'.format(group, name),
        'Source': url,
        'Tracker': '{}/issues'.format(url)}
    author = 'Dominique Benielli and Sokol Koço ' \
             'and Baptiste Bauvin and Cécile Capponi'
    author_email = 'contact.dev@lis-lab.fr'
    license = "BSD-3-Clause"
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS']
    keywords = ('machine learning, supervised learning, classification, '
                'datat generation, multi-view, multi-modal, multi-class')
    packages = find_packages(exclude=['*.tests'])
    install_requires = ['scikit-learn>=0.19', 'numpy>=1.24.0', 'scipy>=1.3.0', "plotly",
                        "h5py", 'pyyaml', 'tabulate', 'pandas', ]
    python_requires = '>=3.5'
    extras_require = {
        'dev': ['pytest', 'pytest-cov'],
        'doc': ['sphinx>=1.8', 'numpydoc', 'sphinx_gallery', 'matplotlib', "jupyter",
                'pandoc', 'nbsphinx', 'nbsphinx_link', 'sphinx_rtd_theme', 'sphinx-autoapi','summit-multi-learn']}
    include_package_data = True

    command_options = {'build_sphinx': {'build_dir':('setup.py', './docs/build/')}}

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
          include_package_data=include_package_data,
          command_options=command_options)


if __name__ == "__main__":
    setup_package()
