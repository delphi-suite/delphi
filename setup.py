from setuptools import find_packages, setup

setup(
    name="delphi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "delphi": ["static/**/*"],
    },
    include_package_data=True,
)
