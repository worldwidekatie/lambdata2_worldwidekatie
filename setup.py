# setup.py file

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lambdata-worldwidekatie", # the name that you will install via pip
    version="1.4",
    author="Katie",
    author_email="katie.evankodouglas@email.com",
    description="This is just a test",
    long_description=long_description,
    long_description_content_type="text/markdown", # required if using a md file for long desc
    license="MIT",
    url="https://github.com/worldwidekatie/lambdata-worldwidekatie",
    #keywords="",
    packages=find_packages() # ["my_lambdata"]
)