from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(name='vllm_add_hyperclovax_model',
      version='0.1',
      packages=find_packages(),
      install_requires=parse_requirements("requirements.txt"),
      entry_points={
        'vllm.general_plugins': [
            "register_hyperclovax_model = model:register",
            "register_hcx_reasoning_parser = parser:register_reasoning_parser",
            "register_hcx_tool_parser = parser:register_tool_parser"
        ]
    }
)
