import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import json
import requests
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import concurrent.futures
import logging
import gc

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置设备  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# GPU 缓存清理函数
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  
        logging.info("GPU cache cleared")
    else:
        logging.info("No GPU available, cache clearing not needed")

# 定义辅助函数
def download_image(item):
    image_url = item['image']
    image_name = os.path.basename(image_url)
    local_path = os.path.join('downloaded_images', image_name)

    if not os.path.exists(local_path):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)  
            return local_path
        except Exception as e:
            logging.error(f"Error downloading {image_url}: {e}") 
            return None
    return local_path

def predownload_images(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs('downloaded_images', exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_image, item) for item in data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc="Downloading images"):
            pass

# 定义数据集类  
class PoleDanceDataset(Dataset):
    def __init__(self, json_file, processor):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_url = item['image']
        image_name = os.path.basename(image_url)
        local_path = os.path.join('downloaded_images', image_name)
        caption = item['caption']

        try:
            image = Image.open(local_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error opening image {local_path}: {e}")
            return None

        try:  
            inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception as e:
            logging.error(f"Error processing image {local_path}: {e}")
            return None

        return inputs

# 定义一个可序列化的collate函数
def collate_fn(batch):  
    return [item for item in batch if item is not None]

# 生成描述函数
def generate_caption(model, processor, image_path):
    model.eval()
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB') 
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=20)  
            generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            return generated_caption
    else:
        logging.warning(f"Image not found: {image_path}")
        return None

# 主程序
def main():
    json_file = 'pole-dance-gestrure-name-data.json'

    logging.info("Pre-downloading images...")
    predownload_images(json_file)

    logging.info("Loading model and processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") 
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)

    logging.info("Creating dataset and dataloader...")
    dataset = PoleDanceDataset(json_file, processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()
    accumulation_steps = 4  # 梯度累积步数

    num_epochs = 60

    # 在训练开始前清理 GPU 缓存
    clear_gpu_cache() 

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            if not batch:  # 跳过空批次
                continue

            input_ids = torch.stack([item['input_ids'] for item in batch]).to(device) 
            pixel_values = torch.stack([item['pixel_values'] for item in batch]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)

            with autocast():
                outputs = model(input_ids=input_ids,  
                                pixel_values=pixel_values,
                                attention_mask=attention_mask,
                                labels=input_ids)

            loss = outputs.loss
            total_loss += loss.item()

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            progress_bar.set_postfix({'loss': loss.item()}) 

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    logging.info("Saving final fine-tuned model and processor...") 
    model.save_pretrained("finetuned_blip_pole_dance")
    processor.save_pretrained("finetuned_blip_pole_dance")

    logging.info("Testing model...")
    model = BlipForConditionalGeneration.from_pretrained("finetuned_blip_pole_dance")
    processor = BlipProcessor.from_pretrained("finetuned_blip_pole_dance")  
    model.to(device)

    test_image_path = os.path.join('downloaded_images', 'speedbump.jpg')
    caption = generate_caption(model, processor, test_image_path)
    if caption:
        logging.info(f"Generated caption: {caption}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred during execution:")