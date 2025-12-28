# HyperCLOVAX vLLM Plugin

Implementation of architectural change on the LLaMA model:
- [μP](https://arxiv.org/pdf/2203.03466) 
- [Peri-LN](https://arxiv.org/pdf/2502.02732)

## Configuration
- [configuration_hyperclovax.py](model/configuration_hyperclovax.py)
  - μP args
    - **embedding_multiplier** (`float`, optional, default: `None`) - Multiplier applied to the embedding weights. If `None`, it is equivalent to `1.0`.
    - **logits_scaling** (`float`, optional, default: `None`) - Scaling factor for logits. If `None`, it is equivalent to `1.0`.
    - **attention_multiplier** (`float`, optional, default: `None`) - Multiplier applied to the attention weights. If `None`, it is equivalent to `self.head_dim ** -0.5`.
    - **residual_multiplier** (`float`, optional, default: `None`) - Scaling factor for residual connections. If `None`, it is equivalent to `1.0`.
  - Peri-LN args
    - **use_post_norm** (`bool`, optional, defaults to `False`) - Determines whether to apply Peri-Layer Normalization. Set to `True` to enable this feature.

## vLLM
- [vllm_hyperclovax.py](model/vllm_hyperclovax.py)
- Reasoning parser: [hcx_reasoner.py](parser/hcx_reasoner.py)
- Tool parser: [hcx_tool_parser.py](parser/hcx_tool_parser.py)

### How to use vLLM ([Docs](https://docs.vllm.ai/en/latest/design/plugin_system.html))
After install vllm, `pip install .` to register `HyperCLOVAXForCausalLM` on vllm package.

### Deploying with vLLM Docker Example ([Docs](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html))
```bash
docker build --tag vllm/vllm-openai-hyperclovax .
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai-hyperclovax <args...>
```

## License
```
HyperCLOVAX vLLM Plugin
Copyright (c) 2025-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
