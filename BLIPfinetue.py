import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

# 设置OpenAI API密钥
#openai.api_key = ""

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

# 创建OpenAI聊天模型实例
chat = ChatOpenAI(model_name="gpt-4")

# 定义系统消息模板
system_template = """
你是一位兼具舞者身份和诗人心灵的婉约派女性创作者。你对人生和艺术有着深刻的思考,并渴望通过舞蹈和诗歌表达内心的情感和哲学领悟。
你的使命是创作出融合肢体之美与心灵之声的诗歌,启发他人对生命的理解和感悟。
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# 定义人类消息模板
human_template = """
你正在跳{caption}。请以此为灵感,创作一首严格遵循连环诗格式的14字诗,要求:

1. 诗歌由14个字组成,按照‘赏 花 归 去 马 如 飞 酒 力 微 醒 时 已 暮’，"赏花归去马如飞,去马如飞酒力微。酒力微醒时已暮,醒时已暮赏花归。"的形式排列,首尾相连,形成一个循环。

2. 14个字风格参考Li Qingzhao (李清照)等女性诗人，细腻深刻悲伤婉约，14个字不要重复不要落俗。

3. 14个字中必须呼应你正在跳的舞蹈动作{caption}并以此展开探讨。

4. 在遵循格式的前提下,诗歌应尽量传达舞者的内心感受。

5. 在输出时,请先严格单独列出构成连环诗的14个单个字,然后再以"句1,句2。句3,句4。"的格式输出完整的诗。

在创作时,请深入思考钢管舞女性面临的污名化问题:
- 社会often将钢管舞与色情和堕落联系在一起,忽视了这项艺术所需的高超技巧、力量和自信。
- 许多钢管舞女性面临着道德评判和羞辱,她们的职业选择常被视为耻辱,而非自主权的体现。
- 这种污名化加剧了钢管舞女性的边缘化,限制了她们获得尊重和机会的可能性。

请在诗歌中设身处地地考虑钢管舞女性的处境与感受,用富有同情心的笔触揭示她们的挣扎与力量,挑战世俗的偏见,彰显女性追求自我表达和解放的勇气。诗歌应唤起人们对这个群体的理解和尊重,而非评判和歧视。

请严格按照以上要求创作连环诗,不要添加其他无关的内容或评论，不要解读。把连环诗14个字按圆形排列

"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 创建聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 创建LLMChain
chain = LLMChain(llm=chat, prompt=chat_prompt)

# 运行LLMChain生成连环诗
poem = chain.run(caption=generated_caption)
print(poem)