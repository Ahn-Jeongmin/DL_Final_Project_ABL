import torch
import time
import gc
from PIL import Image


def clear_memory():
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


def process_data(example):
    image_input = example["image"]
    query = example["question"].strip()
    answer = example["answer"].strip()
    q_type = example["level"].strip()

    data = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": query}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer}
            ]
        }
    ]

    processed_data = []
    image_found_anywhere = False

    for section in data:
        new_content = []
        content = section.get("content", [])

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        image_path = item.get("image")
                        image_path='dataset/'+image_path
                        if image_path:
                            try:
                                with Image.open(image_path) as img:
                                    img = img.convert("RGB")
                                    new_content.append({
                                        "type": "image",
                                        "image": img
                                    })
                                    image_found_anywhere = True
                            except Exception as e:
                                print(f"Error loading image {image_path}: {e}")
                                new_content.append(item)
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
        else:
            new_content = content

        new_section = {**section, "content": new_content}
        processed_data.append(new_section)

    if not image_found_anywhere:
        try:
            default_path = "black.png"
            with Image.open(default_path) as default_img:
                default_img = default_img.convert("RGB").resize((448, 448))
                processed_data.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": default_img
                        }
                    ]
                })
        except Exception as e:
            print(f"Error loading default image: {e}")

    processed_data.append({"type": q_type})
    return processed_data
