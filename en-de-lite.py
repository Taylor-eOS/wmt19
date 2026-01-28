from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import pysbd

print_lim = 100

def main():
    mname = "facebook/wmt19-en-de"
    print(mname)
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    segmenter = pysbd.Segmenter(language="en", clean=False)
    input_file = settings.input_file
    print(f'Opening {input_file}')
    with open(input_file, "r", encoding="utf-8") as infile, open(input_file.replace('in','out'), "w", encoding="utf-8") as outfile:
        line_count = 0
        for raw_line in infile:
            line_count += 1
            line = raw_line.rstrip("\n")
            if not line.strip():
                outfile.write("\n")
                continue
            if settings.print_original: print(f"Original: {line[:print_lim]}{'...' if len(line) > print_lim else ''}")
            sentences = segmenter.segment(line)
            translated = []
            for i, sent in enumerate(sentences, 1):
                if settings.print_original: print(f"{sent[:print_lim]}{'...' if len(sent) > print_lim else ''}")
                input_ids = tokenizer.encode(sent, return_tensors="pt")
                outputs = model.generate(input_ids)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated.append(decoded)
                print(f"{decoded[:print_lim]}{'...' if len(decoded) > print_lim else ''}")
            full_translation = " ".join(translated)
            outfile.write(full_translation + "\n")

if __name__ == "__main__":
    main()

