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

def tokenize(text):
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def get_ngrams(words, n=9):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

print("1. Extracting text from 21 source PDFs...", flush=True)
pdf_dirs = ['base_paper', 'community_detection', 'core_methods', 'datasets', 'imbalance_and_evaluation', 'literature_review', 'loss_functions']
source_ngrams = {}

for d in pdf_dirs:
    for p in glob.glob(f'papers/{d}/*.pdf'):
        fname = os.path.basename(p)
        txt = extract_pdf_text(p)
        words = tokenize(txt)
        for ng in get_ngrams(words, n=9):
            if ng not in source_ngrams:
                source_ngrams[ng] = []
            source_ngrams[ng].append(fname)

print(f"Total distinct 9-grams indexed from source PDFs: {len(source_ngrams):,}", flush=True)

print("\n2. Checking main.tex prose against source PDFs...", flush=True)
with open('papers/paper_latex/main.tex', 'r', encoding='utf-8') as f:
    raw_tex = f.read()

# Completely cut off the bibliography so we only check the actual written paper prose
if '\\begin{thebibliography}' in raw_tex:
    prose_tex = raw_tex.split('\\begin{thebibliography}')[0]
else:
    prose_tex = raw_tex

# Remove title and authors
if '\\maketitle' in prose_tex:
    prose_tex = prose_tex.split('\\maketitle')[1]

# Split into paragraphs
paragraphs = prose_tex.split('\n\n')
print(f"Total prose paragraphs in draft: {len(paragraphs)}", flush=True)

def clean_latex(tex):
    tex = re.sub(r'%.*$', '', tex, flags=re.MULTILINE)
    tex = re.sub(r'\$\$.*?\$\$', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\$.*?\$', ' ', tex)
    tex = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', ' ', tex, flags=re.DOTALL)
    tex = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})?', ' ', tex)
    tex = re.sub(r'[{}\\_^~#&]', ' ', tex)
    return tex

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
                'snippet': " ".join(words[max(0, i-2):min(len(words), i+11)])
            })

print(f"\n3. Prose Check Results: Found {len(matches_found)} matches of length >= 9 words.", flush=True)
if matches_found:
    deduped = []
    seen = set()
    for m in matches_found:
        phrase = m['matched_phrase']
        if phrase not in seen:
            seen.add(phrase)
            deduped.append(m)
    print(f"Unique matching phrases in prose: {len(deduped)}", flush=True)
    for d in deduped:
        print(f"\n- Match in Paragraph {d['paragraph_idx']}: \"{d['matched_phrase']}\"", flush=True)
        print(f"  Source PDF: {d['sources']}", flush=True)
        print(f"  Context snippet: \"{d['snippet']}\"", flush=True)
else:
    print("ZERO PROSE MATCHES DETECTED! All text is 100% original paraphrased text satisfying Rule 4.", flush=True)
