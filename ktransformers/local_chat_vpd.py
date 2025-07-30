"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys
try:
    import torch_gcu # import torch_gcu
    from torch_gcu import transfer_to_gcu # import transfer_to_gcu
except Exception as e:
    print(e)

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate, get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def local_chat(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 300,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = True,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
    chunk_prefill_size: int = 8192
):
    force_think = False
    torch.set_grad_enabled(False)

    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    #if system == "Windows":
    #    os.system("cls")
    #else:
    #    pass
        #os.system("clear")

    content = "强化学习（RL）是一种机器学习（ML）技术，可以训练软件做出决策，以实现最佳结果。它模仿了人类为实现目标所采取的反复试验的学习过程。有助于实现目标的软件操作会得到加强，而偏离目标的操作将被忽略。 RL 算法在处理数据时使用奖惩模式。这些算法从每个操作的反馈中学习，并自行发现实现最终结果的最佳处理路径。此类算法还能够实现延迟满足。最好的整体策略可能需要短期的牺牲，因此其发现的最佳方法可能包括一些惩罚，或在过程中有一些迂回。RL 是一种强大的方法，可以帮助人工智能（AI）系统在看不见的环境中实现最佳结果。使用强化学习（RL）有很多好处。但是，以下三方面往往是最突出的。RL 算法可以在有许多规则和依赖关系的复杂环境中使用。在同一个环境中，即使对环境非常了解，人类可能也无法确定最佳路径。而无模型 RL 算法可以快速适应不断变化的环境，并找到新的策略来优化结果。在传统的 ML 算法中，人类必须通过标记数据对来指导算法。而使用 RL 算法时，就无需手动标记了。这类算法可以自行学习。同时，这类算法还提供整合人类反馈的机制，允许系统根据人类偏好、专业知识和更正进行调试。RL 本质上侧重于长期奖励最大化，因此适用于行动可带来长期后果的场景。它特别适合每一步都无法立即获得反馈的现实情况，因为它可以从延迟的奖励中学习。例如，有关能源消耗或存储的决策可能会产生长期后果。RL 可用于优化长期能源效率和成本。通过适当的架构，RL 代理还可以将学到的策略推广到相似但不相同的任务中。强化学习（RL）可以应用于各种真实用例。下面提供一些示例。在推荐系统等应用场景中，RL 可以根据各个用户的互动情况量身为其推荐内容。这提高了体验的个性化程度。例如，某应用程序可能会根据某些人口统计信息向用户展示广告。该应用程序会通过每次广告互动，了解要向用户展示哪些广告，以改进产品销售情况。传统优化方法通过根据特定标准评估和比较可能的解决方案来解决问题。相比之下，RL 引入了从互动中学习的方式，以便随着时间的推移找到最佳或接近最佳的解决方案。例如，云支出优化系统使用 RL 来适应不断变化的资源需求，并选择最佳实例类型、数量和配置。它根据当前和可用的云基础设施、支出和利用率等因素做出决策。请详细阅读以上文字内容，并概括其大意。"
    print("输入 'exit' 或 'quit' 或按 Ctrl+C 来退出聊天")
    try:
        print(f"Chat: {content}")            
        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if force_think:
            print("============= wrong here ==============")
            token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
            input_tensor = torch.cat(
                [input_tensor, token_thinks], dim=1
            )
        #print(f"============= force_think: {force_think} ==============")
        if mode == 'long_context':
            assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        
        if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
                use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
            )
        else:
            os.system("topsprof --start --session prof_test")
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.GCU,],record_shapes=True,profile_memory=True,with_stack=True,) as prof:
                generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
            )
            #print(prof.key_averages().table(sort_by="self_gcu_time_total", row_limit=40)) # 打印性能统计表格到输出
            prof.export_chrome_trace('trace_S60_v0.2.3_0724.json')
        print("\n推理完成，退出程序...")
        os.system("topsprof --stop --session prof_test")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    fire.Fire(local_chat)
