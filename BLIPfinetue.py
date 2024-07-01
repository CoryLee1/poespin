import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载微调后的模型和处理器
processor = BlipProcessor.from_pretrained("finetuned_blip_pole_dance")
model = BlipForConditionalGeneration.from_pretrained("finetuned_blip_pole_dance")
model.to(device)

# 设置要生成描述的图片路径
image_path = "D:/RWET/poespin/stargazer.jpg"

# 加载并预处理图片
image = Image.open(image_path).convert('RGB')
inputs = processor(images=image, return_tensors="pt").to(device)

# 生成描述
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=20)
    generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated caption: {generated_caption}")