from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

src = "/home/lgene/meu_modelo_temp/model_merged"
dst = "/home/lgene/qwen25_fp16"

model = AutoModelForCausalLM.from_pretrained(
    src,
    dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    src,
    trust_remote_code=True,
    fix_mistral_regex=True
)

model.save_pretrained(dst, safe_serialization=False)
tokenizer.save_pretrained(dst)
