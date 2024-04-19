from setuptools import find_packages, setup

setup(
    name="delphi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "delphi": ["test_configs/**/*"],
    },
    include_package_data=True,
)
