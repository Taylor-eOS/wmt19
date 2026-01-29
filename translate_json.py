from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import pysbd
import settings
import json

def main():
    print(f'Model: {settings.mname}')
    tokenizer = FSMTTokenizer.from_pretrained(settings.mname)
    model = FSMTForConditionalGeneration.from_pretrained(settings.mname)
    segmenter = pysbd.Segmenter(language="en", clean=False)
    input_file = settings.input_file.replace('txt','json')
    print(f'Opening {input_file}')
    lim = settings.print_lim
    with open(input_file, "r", encoding="utf-8") as infile, open(input_file.replace('in','out'), "w", encoding="utf-8") as outfile:
        line_count = 0
        for raw_line in infile:
            line_count += 1
            line = raw_line.rstrip("\n")
            if not line.strip():
                outfile.write("\n")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                outfile.write(raw_line)
                continue
            if "text" not in obj or not isinstance(obj["text"], str):
                outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            text = obj["text"]
            if settings.print_original:
                print(f"Original: {text[:lim]}{'...' if len(text) > lim else ''}")
            sentences = segmenter.segment(text)
            translated = []
            for sent in sentences:
                if settings.print_original:
                    print(f"{sent[:lim]}{'...' if len(sent) > lim else ''}")
                if not sent.strip():
                    translated.append("")
                    continue
                input_ids = tokenizer.encode(sent, return_tensors="pt")
                outputs = model.generate(input_ids)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated.append(decoded)
                print(f"{decoded[:lim]}{'...' if len(decoded) > lim else ''}")
            full_translation = " ".join(translated)
            obj["text"] = full_translation
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

