from vllm import ModelRegistry

def register():
    from .vllm_hyperclovax import HyperCLOVAXForCausalLM

    if "HyperCLOVAXForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("HyperCLOVAXForCausalLM", HyperCLOVAXForCausalLM)
