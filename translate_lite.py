from transformers import FSMTForConditionalGeneration, FSMTTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import pysbd
import settings

def main():
    print(f'Model: {settings.mname}')
    tokenizer = AutoTokenizer.from_pretrained(settings.mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.mname)
    segmenter = pysbd.Segmenter(language=settings.segmenter_language, clean=False)
    input_file = settings.input_file
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
            if settings.print_original: print(f"Original: {line[:lim]}{'...' if len(line) > lim else ''}")
            sentences = segmenter.segment(line)
            translated = []
            for i, sent in enumerate(sentences, 1):
                if settings.print_original: print(f"{sent[:lim]}{'...' if len(sent) > lim else ''}")
                input_ids = tokenizer.encode(sent, return_tensors="pt")
                outputs = model.generate(input_ids)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated.append(decoded)
                print(f"{decoded[:lim]}{'...' if len(decoded) > lim else ''}")
            full_translation = " ".join(translated)
            outfile.write(full_translation + "\n")

if __name__ == "__main__":
    main()

