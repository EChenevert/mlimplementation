from setuptools import setup

setup(name='mlimplementation',
      version='0.1',
      description='function to help in ml analysis',
      author='Etienne Chenevert',
      author_email='etachen@iu.edu',
      packages=['mlimplementation'],
      # package_data={"": ["DATA/*.csv"]},
      install_requires=['scikit-learn']
)