import torch

def quantize_weights(w: torch.Tensor, num_bits=4, group_size=128):
    orig_device = w.device
    size_k, size_n = w.shape
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2

    # Reshape to [groupsize, -1]
    if group_size < size_k:
        w = w.view((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / max_q_val  # 2 => symmetric

    # Quantize
    q_w = torch.round(w / s).int()
    q_w += half_q_val
    q_w = torch.clamp(q_w, 0, max_q_val)

    # Restore original shapes
    if group_size < size_k:
        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        q_w = reshape_w(q_w)

    s = s.reshape((-1, size_n)).contiguous()
    return (
        q_w.to(device=orig_device),
        s.to(device=orig_device),
    )
    

# input:  weight shape: [N, K] bfloat16
# output: rweight [K/2, N] uint8,  deq_zeros [K/group_size, N] bfloat16, scales [K/group_size, N] bfloat16
def rearrange_float16_weight(
    original_weight, rearrange_group=128
):
    # if sym_quant:
    #    qweight, qzeros, scales = _quant_weight_sym(original_weight, rearrange_group)
    # else:
    #    qweight, qzeros, scales = _quant_weight_asym(original_weight, rearrange_group)
    qweight, scales = quantize_weights(original_weight.t().contiguous())
    deq_zeros = 8.0 * scales

        
    # weight rearrange
    if qweight.shape[0] % rearrange_group != 0:
        padding = torch.zeros(
            [qweight.shape[0] % rearrange_group, qweight.shape[1]],
            dtype=qweight.dtype,
            device=qweight.device,
        )
        qweight = torch.concat([qweight, padding], dim=0)
    rweight_shape = (int(qweight.shape[0] / 2), qweight.shape[1])
    rweight = torch.zeros(rweight_shape, dtype=torch.uint8).to(qweight.device)
    half_group = int(rearrange_group / 2)
    try:
        shifts = torch.arange(0, qweight.shape[0], device=qweight.device).reshape(
            int(qweight.shape[0] / half_group), -1
        )
        rweight |= torch.bitwise_left_shift(qweight[shifts[::2].reshape(-1)], 0)
        rweight |= torch.bitwise_left_shift(qweight[shifts[1::2].reshape(-1)], 4)
    except Exception as e:
        raise RuntimeError(f"weight rearrange error: {e}")

    return rweight, deq_zeros.to(torch.bfloat16), scales.to(torch.bfloat16)

# from safetensors.torch import load_file
# W_FN = "/data/LLM/deepseek_r1_pretrained_bf16/model-00001-of-000163.safetensors"
# weight_dict = load_file(W_FN)
# test_weight = weight_dict["model.layers.0.self_attn.o_proj.weight"]
# rweight, deq_zeros, scales = rearrange_float16_weight(test_weight, True)
# pass 