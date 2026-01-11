from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 假设你的模型是类似 Llama 或 GPT 的架构
model_name_or_path = r"C:\Users\Administrator\Downloads\d4"  # 你的文件夹路径

# 加载 tokenizer（需要你有 tokenizer 文件，比如 tokenizer.json 或 vocab.txt）
# 如果你没有，可能需要从原始模型下载（见下文）
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 加载模型（这里会自动找 model_*.pt 和 config.json 等）
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,  # 可选，节省显存
    device_map="auto",          # 自动分配到 GPU/CPU
)

# 设置为推理模式
model.eval()

# 测试输入
input_text = "who are u?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# 生成回复
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型回复：", response)