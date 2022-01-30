from setuptools import setup

setup(
   name='sme',
   version='0.1',
   author='Ander Alonso',
   author_email='aalonso213@ikasle.ehu.eus',
   packages=['sme'],
   url='[Indicar una URL para el paquete...]',
   license='LICENSE.txt',
   description='Trabajo final software matematico y estadistico',
   long_description=open('README.md').read(),
   install_requires=[
      "seaborn >= 0.9.0", 
      "pandas >= 0.25.1", 
      "plotnine >= 0.6.0", 
      "numpy >=1.17.2",
      "sklearn"
   ],
)
