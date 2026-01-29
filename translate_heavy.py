import os
import torch
from transformers import MarianMTModel, MarianTokenizer
import settings

def check_backends():
    try:
        import sentencepiece
    except Exception as e:
        raise RuntimeError("sentencepiece is required for MarianTokenizer") from e

def load_model_and_tokenizer(model_name, device):
    print("Checking backends...")
    check_backends()
    print("Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Model ready on", device)
    return model, tokenizer

def translate_text(text, model, tokenizer, device):
    if not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=512, num_beams=4)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def process_file(input_path, output_path, model, tokenizer, device):
    print("Reading from", input_path)
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for i, raw_line in enumerate(infile, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                outfile.write("\n")
                continue
            out = translate_text(line, model, tokenizer, device)
            print(i, out)
            outfile.write(out + "\n")
            outfile.flush()

def main():
    input_path = settings.input_file
    output_path = settings.output_file

    if not os.path.isfile(input_path):
        raise RuntimeError("Input file does not exist: " + input_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Helsinki-NLP/opus-mt-en-de"

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    process_file(input_path, output_path, model, tokenizer, device)
    print("Done.")

if __name__ == "__main__":
    main()

