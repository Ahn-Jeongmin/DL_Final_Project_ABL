from openai import OpenAI
import os
import base64
import json
from prompt import entity, unamb_entity
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

client = OpenAI(
    api_key='api-key'
)


def process_image(line: str):
    try:
        json_data = json.loads(line)
        vis_image_path = json_data['visualization_path'].replace('output_vis/', '')

        with open(vis_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = unamb_entity.format(boxes=json_data['count_label'])

        response = client.responses.create(
            model="gpt-5-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded_image}"},
                ],
            }],
        )

        json_data['entity_description'] = response.output_text
        return json_data

    except Exception as e:
        return {"error": str(e)}


def main():
    with open('test_unambig_annotations.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_output = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, line) for line in lines]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing entities", unit="image"):
            result = future.result()
            data_output.append(result)

    with open('test_unambig_entity_descriptions.jsonl', 'w', encoding='utf-8') as f:
        for item in data_output:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


if __name__ == "__main__":
    main()
