# Global imports
import setuptools

# package name: osr = Object Search Research
PACKAGE = 'osr'

# Setup function
setuptools.setup(
    name='{}-lib'.format(PACKAGE),
    namespace_packages=[PACKAGE],
    version=open('VERSION').read().strip(),
    description='Object Search Research Library',
    packages=['osr', 'osr.data', 'osr.models', 'osr.engine', 'osr.losses',
        'osr.viz', 'osr.plot'],
    package_dir={'osr': 'src'},
    entry_points={
        'console_scripts': [
            (
                '{pkg}_prep_cuhk = '
                '{pkg}.data.cuhk_utils:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_prep_prw = '
                '{pkg}.data.prw_utils:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_prep_cocopersons = '
                '{pkg}.data.cocopersons_utils:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_crop_cocopersons = '
                '{pkg}.data.cocopersons_utils:crop_cocop'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_run = '
                '{pkg}.engine.main:main'
                .format(pkg=PACKAGE)
            ),
        ]
    }
) 
