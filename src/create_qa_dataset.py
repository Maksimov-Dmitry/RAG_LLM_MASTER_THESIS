import click
import os
from tqdm import tqdm
import requests
import base64
from dotenv import load_dotenv
import json
from datasets import load_dataset, DatasetDict


def get_dataset(file_path, page, document):
    ds = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if '{' in line and '}' in line:
                dict_str = line[line.find('{') : line.rfind('}') + 1].replace('\\', '')
                try:
                    data = json.loads(dict_str)
                    question, ground_truth = next(iter(data.items()))
                    ds.append({"question": question, "ground_truth": ground_truth, "page": int(page), "document": document})
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in line: {repr(line)}")
    return ds


def extract_version_number(folder_name):
    try:
        return int(folder_name.split('_')[0])
    except ValueError:
        return -1


def get_questions(image, num_questions_per_chunk):
    prompt_template = f"""
    Geben Sie die Bildinformationen und kein Vorwissen an. Generieren Sie nur Fragen, die auf der untenstehenden Abfrage basieren.

    Sie sind ein Lehrer/Professor. Ihre Aufgabe ist es, {num_questions_per_chunk} Fragen für ein bevorstehendes Quiz/eine Prüfung zu erstellen.
    Die Fragen sollten sich auf das Bild beziehen. Fragen Sie nicht, was auf dem Bild zu sehen ist, da es in der Prüfung kein Diagramm geben wird. Beschränken Sie die Fragen auf die bereitgestellten Kontextinformationen.
    Wenn Sie meinen, dass {num_questions_per_chunk} Fragen zu viel für dieses Bild sind, können Sie die Anzahl der Fragen reduzieren.
    Bitte geben Sie die Antwort im JSON-Format zurück:
    [{{Frage: Antwort}}, {{Frage: Antwort}}]
    Beispiel:
    [
    {{"Was beschreibt der Begriff 'Headroom' im Kontext der Crimp-Kontaktierung?": "Der Abstand des gemittelten Spitzenwerts der Gut-Crimps und des gemittelten Spitzenwerts der Leer-Crimps."}},
    {{"Wie wird die Streuung der Spitzenwerte bei Gut-Crimps gemessen?": "Als Standardabweichung der Messwerte relativ zum gemittelten Spitzenwert, in Prozent ausgedrückt."}},
    {{"Was ist der Unterschied zwischen den Mittelwerten der Crimp-Kraftkurve von Gut-Crimp und Leer-Crimp?": "Der Mittelwert der Crimp-Kraftkurve von Gut-Crimp ist der durchschnittliche Wert der guten Crimps, während der Mittelwert der Crimp-Kraftkurve von Leer-Crimp der Durchschnittswert der Leer-Crimps ohne Draht ist."}},
    {{"Welche Information liefert der maximale Kraftwert (Fmax) einer Crimp-Kraftkurve?": "Er gibt die höchste Kraft an, die während des Crimp-Vorgangs aufgezeichnet wurde."}},
    {{"Was könnte ein hoher Headroom-Wert bei der Crimp-Kontaktierung andeuten?": "Ein hoher Headroom-Wert könnte auf Probleme wie fehlende Drähte oder Isolationsfehler im Crimp hinweisen."}}
    ]
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_template
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.85,
        "top_p": 0.9,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@click.command()
@click.option('--input_directory', default='data/images', help='Directory with images.')
@click.option('--output_directory', default='data/qa', help='Directory for the dataset.')
@click.option('--num_questions_per_chunk', default=8, help='Number of questions per chunk.')
@click.option('--qa_dataset_ouput', default='data/processed', help='Path to the output dataset.')
@click.option('--train_size', default=0.7, help='Train size.')
def create_qa_dataset(input_directory, output_directory, num_questions_per_chunk, qa_dataset_ouput, train_size):
    ds = []
    if not os.path.exists(input_directory):
        print(f"Input directory {input_directory} does not exist.")
        return
    load_dotenv()
    os.makedirs(output_directory, exist_ok=True)
    folders = os.listdir(output_directory)
    version_numbers = [extract_version_number(folder) for folder in folders]
    max_version_number = max(version_numbers) if version_numbers else -1
    output_directory = os.path.join(output_directory, f'{max_version_number + 1}')
    os.makedirs(output_directory, exist_ok=True)

    for document in tqdm(os.listdir(input_directory)):
        dir_path = os.path.join(input_directory, document)
        if os.path.isdir(dir_path):
            out_doc_dir = os.path.join(output_directory, document)
            os.makedirs(out_doc_dir, exist_ok=True)
            for page in os.listdir(dir_path):
                if page.endswith(".png"):
                    image = encode_image(os.path.join(dir_path, page))
                    response = get_questions(image, num_questions_per_chunk)
                    with open(f'{out_doc_dir}/{page.replace(".png", ".txt")}', 'w') as f:
                        f.write(response['choices'][0]['message']['content'])
                    ds.extend(get_dataset(f'{out_doc_dir}/{page.replace(".png", ".txt")}', page.replace(".png", ''), document))
    with open(f'{qa_dataset_ouput}/data.json', 'w') as f:
        for d in ds:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

    dataset = load_dataset('json', data_files=f'{qa_dataset_ouput}/data.json', split='train')
    train_testvalid = dataset.train_test_split(train_size=train_size, shuffle=True, seed=42)
    test_valid = train_testvalid['test'].train_test_split(train_size=0.5, shuffle=False)
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'val': test_valid['train']})
    train_test_valid_dataset.save_to_disk(f'{qa_dataset_ouput}/dataset')


if __name__ == '__main__':
    create_qa_dataset()
