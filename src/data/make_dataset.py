import click
import os
import fitz
from tqdm import tqdm
import io
import shutil
from PIL import Image


def _extract_text(block):
    text = ""
    for line in block["lines"]:
        for span in line["spans"]:
            text += span["text"] + " "
    return text.strip()


@click.command()
@click.option('--input_directory', default='data/raw', help='Directory with the source PDF files.')
@click.option('--output_directory', default='data/interim', help='Directory for the transformed files.')
def process_pdfs(input_directory, output_directory):
    if not os.path.exists(input_directory):
        print(f"Input directory {input_directory} does not exist.")
        return
    shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    for filename in tqdm(os.listdir(input_directory)):
        if filename.endswith(".pdf"):
            input_file_path = os.path.join(input_directory, filename)
            output_directory_path = os.path.join(output_directory, filename.replace(".pdf", ""))
            os.makedirs(output_directory_path, exist_ok=True)
            doc = fitz.open(input_file_path)
            for page_id, page in enumerate(doc):
                try:
                    tables = page.find_tables()
                    tables_len = len(tables.tables)
                except IndexError:
                    tables = []
                    tables_len = 0
                    print(f"Error in {filename} page {page_id}")
                object = page.get_text("dict")

                count = 0
                i_table = 0
                for block in object["blocks"]:
                    if (tables_len > 0 and tables_len > i_table
                            and block['bbox'][1] >= tables[i_table].bbox[1] and block['bbox'][3] <= tables[i_table].bbox[3]):
                        text = ""
                        for line in tables[i_table].extract():
                            text += str(line) + "\n"
                        with open(f'{output_directory_path}/{page_id}_{count}_table.txt', 'w') as f:
                            f.write(text)
                            count += 1
                            i_table += 1
                    if block["type"] == 0:
                        text = _extract_text(block)
                        if len(text) > 0:
                            with open(f'{output_directory_path}/{page_id}_{count}_text.txt', 'w') as f:
                                f.write(text)
                                count += 1
                    if block["type"] == 1:
                        img = Image.open(io.BytesIO(block['image']))
                        try:
                            img.save(f'{output_directory_path}/{page_id}_{count}_image.png')
                            count += 1
                        except OSError:
                            print(f"Error in {filename} page {page_id} block {count}")


if __name__ == '__main__':
    process_pdfs()
