import lmstudio as lms
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Captioning-модель + генератор
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

llm_model = lms.llm("yandexgpt-5-lite-8b-instruct")

def answer_question(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_length=100)
    answer = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return answer

while True:
    print("\nВведите путь к изображению")
    #image_input = input().strip()
    image_input = 'image.jpg'
    print("\nВведите запрос")
    user_input = input()

    try:
        answer = answer_question(image_input)
        print(f"\n[BLIP ответ]: {answer}")

        task = f"Отвечай на том языке на котором задан вопрос. Твоими ограничениями является окружение которое ты видишь. Если чего-то нет в наличии об этом говоришь Ты видишь: {answer}. Задача: {user_input}."
        response = llm_model.respond(task)
        print(f"\n[Ответ LLM]: {response}")

    except Exception as e:
        print("Ошибка:", e)