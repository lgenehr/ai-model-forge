from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/home/lgene/meu_modelo_temp/model_merged",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "/home/lgene/meu_modelo_temp/model_merged",
    trust_remote_code=True,
    fix_mistral_regex=True
)

inputs = tokenizer("Explique atenção em transformers", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0], skip_special_tokens=True))