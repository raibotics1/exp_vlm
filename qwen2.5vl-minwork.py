import lmstudio as lms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

llm_model = lms.llm("yandexgpt-5-lite-8b-instruct")

def get_objects_description(image_path: str) -> str:
    """
    Запрос к Qwen 2.5 VL: 
    - подробное описание объектов,
    - их количество,
    - координаты в относительном формате (от 0 до 1).
    Без передачи пользовательского вопроса.
    """
    prompt_text = (
        f"Describe the image in detail. List all objects, their count, and coordinates.\n" 
        f"Use the format: object - quantity - coordinates. \n"
        f"Coordinates must be in relative format (values between 0 and 1)."

    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=1000)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


if __name__ == "__main__":
    print("\nВведите путь к изображению (локальный путь или URL):")
    image_path = input().strip()

    try:
        description = get_objects_description(image_path)
        print("\n[Описание объектов от Qwen 2.5 VL]:")
        print(description)

        print("\nВведите свой вопрос для LLM:")
        user_question = input().strip()

        llm_prompt = (
            f"Отвечай на том языке, на котором задан вопрос.\n"
            f"Ты видишь следующее описание окружения:\n{description}\n"
            f"Задача: {user_question}\n"
            f"Если не можешь выполнить задачу ответь что не можешь и объясни почему. Можешь предложить решение. \n"
            f"Если окружение ты не видишь, то обычный чат.\n"
            f"Если окружение на другом языке переведи на язык  на котором написана задача"
        )
        llm_response = llm_model.respond(llm_prompt)
        print(f"\n[Ответ LLM]: {llm_response}")

    except Exception as e:
        print(f"\nОшибка: {e}")
