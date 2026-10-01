import os, glob, re, sys
import pypdf

def extract_pdf_text(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt:
                full_text.append(txt)
        return " ".join(full_text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def clean_latex(tex):
    # Remove comments
    tex = re.sub(r'%.*$', '', tex, flags=re.MULTILINE)
    # Remove equations
    tex = re.sub(r'\$\$.*?\$\$', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\$.*?\$', ' ', tex)
    tex = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', ' ', tex, flags=re.DOTALL)
    # Remove environments like tabular, thebibliography
    tex = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', ' ', tex, flags=re.DOTALL)
    # Remove LaTeX commands \command{arg} or \command
    tex = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})?', ' ', tex)
    # Remove special characters
    tex = re.sub(r'[{}\\_^~#&]', ' ', tex)
    return tex

def tokenize(text):
    # Normalize whitespace and non-alphanumeric
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return words

def get_ngrams(words, n=9):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

print("1. Extracting text from source PDFs...")
pdf_dirs = ['base_paper', 'community_detection', 'core_methods', 'datasets', 'imbalance_and_evaluation', 'literature_review', 'loss_functions']
source_ngrams = {} # ngram -> list of (pdf_name, index)

for d in pdf_dirs:
    for p in glob.glob(f'papers/{d}/*.pdf'):
        fname = os.path.basename(p)
        print(f"   Extracting: {fname}")
        txt = extract_pdf_text(p)
        words = tokenize(txt)
        ngrams = get_ngrams(words, n=9)
        for ng in ngrams:
            if ng not in source_ngrams:
                source_ngrams[ng] = []
            source_ngrams[ng].append(fname)

print(f"Total distinct 9-grams indexed from source PDFs: {len(source_ngrams):,}")

print("\n2. Checking main.tex against source PDFs...")
with open('papers/paper_latex/main.tex', 'r', encoding='utf-8') as f:
    raw_tex = f.read()

# Split into paragraphs
paragraphs = raw_tex.split('\n\n')
print(f"Total paragraphs in draft: {len(paragraphs)}")

matches_found = []

for p_idx, p in enumerate(paragraphs):
    cleaned = clean_latex(p)
    words = tokenize(cleaned)
    if len(words) < 9:
        continue
    p_ngrams = get_ngrams(words, n=9)
    for i, ng in enumerate(p_ngrams):
        if ng in source_ngrams:
            matched_phrase = " ".join(ng)
            sources = list(set(source_ngrams[ng]))
            matches_found.append({
                'paragraph_idx': p_idx,
                'matched_phrase': matched_phrase,
                'sources': sources,
                'words': words[max(0, i-2):min(len(words), i+11)]
            })

print(f"\n3. Results: Found {len(matches_found)} matches of length >= 9 words.")
if matches_found:
    # Deduplicate consecutive overlapping ngrams
    deduped = []
    seen = set()
    for m in matches_found:
        phrase = m['matched_phrase']
        if phrase not in seen:
            seen.add(phrase)
            deduped.append(m)
    print(f"Unique matching phrases: {len(deduped)}")
    for d in deduped[:20]:
        print(f"\n- Match: \"{d['matched_phrase']}\"")
        print(f"  Sources: {d['sources']}")
else:
    print("NO OVERLAP DETECTED! All text strictly satisfies the <=8 word threshold.")
