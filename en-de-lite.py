from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import pysbd

def main():
    mname = "facebook/wmt19-en-de"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    segmenter = pysbd.Segmenter(language="en", clean=False)
    with open("input.txt", "r", encoding="utf-8") as infile, open("output.txt", "w", encoding="utf-8") as outfile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")
            if not line.strip():
                outfile.write("\n")
                continue
            sentences = segmenter.segment(line)
            translated = []
            for sent in sentences:
                input_ids = tokenizer.encode(sent, return_tensors="pt")
                outputs = model.generate(input_ids)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated.append(decoded)
            outfile.write(" ".join(translated) + "\n")

if __name__ == "__main__":
    main()

