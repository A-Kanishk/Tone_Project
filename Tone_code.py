# Full code: Tone, Readability + FinBERT Contextual Sentiment Analysis
# Adapted for CodeOcean platform

# Step 1: Install packages
import sys
import subprocess

def install_packages():
    """Install required packages in CodeOcean environment"""
    packages = ['nltk', 'pandas', 'openpyxl', 'transformers', 'torch', 'pypdf']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# Step 2: Import libraries
import os
import re
import math
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
from pypdf import PdfReader

warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("✅ All packages loaded successfully")

# Step 3: Set up data directories (CodeOcean structure)
# CodeOcean uses /data for input data and /results for outputs
DATA_DIR = '/data'  # Input data directory in CodeOcean
RESULTS_DIR = '/results'  # Output directory in CodeOcean

# For local testing, use current directory
if not os.path.exists(DATA_DIR):
    DATA_DIR = './data'
    RESULTS_DIR = './results'
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Results directory: {RESULTS_DIR}")

# Step 4: Load FinBERT model for contextual financial sentiment
print("\n🤖 Loading FinBERT model for contextual sentiment analysis...")
print("This model understands financial context (e.g., 'challenging market' = negative)")

finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_model.eval()  # Set to evaluation mode

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
finbert_model.to(device)
print(f"✅ FinBERT loaded successfully on {device}")

# Step 5: Load Loughran-McDonald Dictionary
# In CodeOcean, place lm_dictionary.csv in /data directory
dictionary_path = os.path.join(DATA_DIR, 'lm_dictionary.csv')

if not os.path.exists(dictionary_path):
    print("\n⚠️  WARNING: lm_dictionary.csv not found!")
    print(f"Please upload lm_dictionary.csv to {DATA_DIR}")
    print("You can download it from: https://drive.google.com/file/d/1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY")
    # Try to download if gdown is available
    try:
        import gdown
        gdown.download(id='1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY', 
                      output=dictionary_path, quiet=False)
    except:
        raise FileNotFoundError(f"Dictionary file not found at {dictionary_path}")

dictionary = pd.read_csv(dictionary_path)
dictionary['Word'] = dictionary['Word'].str.lower()

# Handle different possible column names
if 'Weak_Modal' in dictionary.columns:
    modal_weak_col = 'Weak_Modal'
elif 'WeakModal' in dictionary.columns:
    modal_weak_col = 'WeakModal'
elif 'ModalWeak' in dictionary.columns:
    modal_weak_col = 'ModalWeak'
else:
    raise KeyError("Modal column not found in dictionary.")

positive_words = set(dictionary[dictionary['Positive'] > 0]['Word'].tolist())
negative_words = set(dictionary[dictionary['Negative'] > 0]['Word'].tolist())
uncertain_words = set(dictionary[dictionary['Uncertainty'] > 0]['Word'].tolist())
weak_modal_words = set(dictionary[dictionary[modal_weak_col] > 0]['Word'].tolist())
print("✅ Dictionary word lists extracted successfully.")

# Step 6: Syllable count function
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if len(word) == 0:
        return 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count += 1
    return count

# Step 7: Extract year and quarter from filenames
def extract_year_quarter(filename):
    # For transcript files like '500112_2020_Q3.txt'
    year_quarter_match = re.search(r'(\d{4})_Q([1-4])', filename)
    if year_quarter_match:
        return year_quarter_match.group(1), f"Q{year_quarter_match.group(2)}"

    quarter_year_match = re.search(r'Q([1-4])_(\d{4})', filename)
    if quarter_year_match:
        return quarter_year_match.group(2), f"Q{quarter_year_match.group(1)}"

    # For annual report files like '500112_2000.pdf'
    year_match = re.search(r'_(\d{4})\.', filename)
    if year_match:
        return year_match.group(1), "Annual"

    # Fallback for other formats
    year_match = re.search(r'\d{4}', filename)
    year = year_match.group(0) if year_match else 'Unknown'
    quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
    quarter = f"Q{quarter_match.group(1)}" if quarter_match else 'Unknown'

    # If it's a year-only match and no quarter, label as Annual
    if quarter == 'Unknown' and year != 'Unknown':
        return year, "Annual"

    return year, quarter


# Step 8: FinBERT Contextual Sentiment Analysis
def analyze_finbert_sentiment(text, max_sentences=50):
    """
    Analyze sentiment using FinBERT (contextual understanding)
    FinBERT understands financial context better than word counting

    Returns: positive_score, negative_score, neutral_score (0-1)
    """
    sentences = sent_tokenize(text)

    # Limit to first N sentences for performance (adjust as needed)
    sentences = sentences[:max_sentences]

    if len(sentences) == 0:
        return 0, 0, 0

    positive_scores = []
    negative_scores = []
    neutral_scores = []

    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 5:
            continue

        # Tokenize and get sentiment
        inputs = finbert_tokenizer(sentence, return_tensors="pt",
                                   truncation=True, max_length=512,
                                   padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = finbert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [positive, negative, neutral]
        pos_score = predictions[0][0].item()
        neg_score = predictions[0][1].item()
        neu_score = predictions[0][2].item()

        positive_scores.append(pos_score)
        negative_scores.append(neg_score)
        neutral_scores.append(neu_score)

    # Average across all sentences
    avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0
    avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0

    return avg_positive, avg_negative, avg_neutral

# Step 9: PDF text extraction function
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"  Could not read PDF {os.path.basename(pdf_path)}: {e}")
    return text

# Step 10: Main analysis function
def analyze_text(text, file_path):
    try:
        # Traditional analysis (Loughran-McDonald word counting)
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        sentences = sent_tokenize(text)
        words = word_tokenize(clean_text)
        total_words = len(words)
        total_sentences = len(sentences) if len(sentences) > 0 else 1

        # Word-based counts (traditional method)
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        uncertain_count = sum(1 for word in words if word in uncertain_words)
        weak_modal_count = sum(1 for word in words if word in weak_modal_words)

        # Traditional tone score (word counting)
        traditional_tone = (positive_count - negative_count) / (positive_count + negative_count + 1e-10)

        # Ambiguity percentages
        percent_uncertain = (uncertain_count / total_words) * 100 if total_words > 0 else 0
        percent_weak_modal = (weak_modal_count / total_words) * 100 if total_words > 0 else 0

        # Readability
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        file_size_readability = -math.log(size_mb + 1)

        if total_words > 0 and total_sentences > 0:
            avg_words_per_sentence = total_words / total_sentences
            complex_words = [w for w in words if syllable_count(w) >= 3]
            percent_complex = (len(complex_words) / total_words) * 100
            fog_index = 0.4 * (avg_words_per_sentence + percent_complex)
        else:
            fog_index = 0

        # FinBERT Contextual Sentiment (NEW!)
        print("  Running FinBERT contextual analysis...")
        finbert_pos, finbert_neg, finbert_neu = analyze_finbert_sentiment(text, max_sentences=50)

        # FinBERT-based tone score (positive - negative)
        finbert_tone = finbert_pos - finbert_neg

        return {
            # Traditional metrics
            'Positive_Count': positive_count,
            'Negative_Count': negative_count,
            'Uncertain_Count': uncertain_count,
            'Weak_Modal_Count': weak_modal_count,
            'Traditional_Tone_Score': traditional_tone,
            'Percent_Uncertain': percent_uncertain,
            'Percent_Weak_Modal': percent_weak_modal,

            # FinBERT contextual metrics (NEW!)
            'FinBERT_Positive': finbert_pos,
            'FinBERT_Negative': finbert_neg,
            'FinBERT_Neutral': finbert_neu,
            'FinBERT_Tone_Score': finbert_tone,

            # Readability
            'Total_Words': total_words,
            'Total_Sentences': total_sentences,
            'File_Size_MB': size_mb,
            'File_Size_Readability': file_size_readability,
            'Fog_Index': fog_index
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Step 11: Main processing loop for transcripts and annual reports
def process_files(root_dir, file_extension, is_pdf=False):
    results = []
    
    if not os.path.exists(root_dir):
        print(f"⚠️  Directory not found: {root_dir}")
        return results
    
    company_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    total_companies = len(company_folders)
    print(f"\nTotal companies to process in {os.path.basename(root_dir)}: {total_companies}\n")

    file_count = 0
    for index, company_folder in enumerate(company_folders, 1):
        company_path = os.path.join(root_dir, company_folder)
        if os.path.isdir(company_path):
            company_id = company_folder
            print(f"\n{'='*60}")
            print(f"Company {company_id} ({index}/{total_companies})")
            print(f"{'='*60}")

            for filename in os.listdir(company_path):
                if filename.endswith(file_extension):
                    file_path = os.path.join(company_path, filename)
                    print(f"Analyzing: {filename}")

                    text = ""
                    if is_pdf:
                        text = extract_text_from_pdf(file_path)
                    else:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()

                    if not text:
                        print(f"  No text extracted from {filename}. Skipping.")
                        continue

                    year, quarter = extract_year_quarter(filename)
                    analysis = analyze_text(text, file_path)

                    if analysis:
                        analysis['Company_ID'] = company_id
                        analysis['Year'] = year
                        analysis['Quarter'] = quarter
                        analysis['File_Name'] = filename
                        results.append(analysis)
                        file_count += 1

                        if file_count % 25 == 0:
                            print(f"\n📊 Progress: {file_count} files analyzed in this category")

    return results

# Step 12: Analysis and insights function
def create_comparison_insights(df, report_type):
    """Compare traditional word-counting vs FinBERT contextual analysis"""
    insights = []
    insights.append("="*80)
    insights.append(f"COMPARISON FOR: {report_type}")
    insights.append("="*80)
    insights.append("")
    insights.append(f"Total documents analyzed: {len(df):,}")
    insights.append(f"Unique companies: {df['Company_ID'].nunique():,}")
    insights.append(f"Time period: {df['Year'].min()} - {df['Year'].max()}")
    insights.append("")
    insights.append("TRADITIONAL METHOD (Word Counting):")
    insights.append(f"  Average tone: {df['Traditional_Tone_Score'].mean():.4f}")
    insights.append("")
    insights.append("FINBERT METHOD (Contextual Understanding):")
    insights.append(f"  Average tone: {df['FinBERT_Tone_Score'].mean():.4f}")
    insights.append("")
    corr = df['Traditional_Tone_Score'].corr(df['FinBERT_Tone_Score'])
    insights.append(f"CORRELATION between methods: {corr:.4f}")
    insights.append("")
    insights.append("READABILITY ANALYSIS:")
    insights.append(f"  Average Fog Index: {df['Fog_Index'].mean():.2f}")
    insights.append(f"  Average document length: {df['Total_Words'].mean():,.0f} words")
    insights.append("")
    return "\n".join(insights)

# Step 13: Run Analysis and Save Results
print("\n" + "="*80)
print("📊 STARTING ANALYSIS")
print("="*80)

# Define directories for annual reports and transcripts
# In CodeOcean, organize your data as:
# /data/annual_report/[company_folders]/[pdf_files]
# /data/scripts_text/[company_folders]/[txt_files]

annual_reports_dir = os.path.join(DATA_DIR, 'annual_report')
transcripts_dir = os.path.join(DATA_DIR, 'scripts_text')

# Process Annual Reports (PDFs)
print("\n📄 Processing Annual Reports...")
annual_report_results = process_files(annual_reports_dir, '.pdf', is_pdf=True)

# Process Transcripts (TXTs)
print("\n📝 Processing Quarterly Transcripts...")
transcript_results = process_files(transcripts_dir, '.txt', is_pdf=False)

# Step 14: Save results to Excel with two sheets
output_path = os.path.join(RESULTS_DIR, 'combined_analysis_with_finbert.xlsx')

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    if annual_report_results:
        df_annual = pd.DataFrame(annual_report_results)
        df_annual = df_annual.sort_values(['Company_ID', 'Year'])
        df_annual.to_excel(writer, sheet_name='Annual_Reports', index=False)
        print("\n✅ Annual reports data prepared for Excel.")
        
        # Print insights
        insights = create_comparison_insights(df_annual, "Annual Reports")
        print("\n" + insights)
    else:
        print("\n⚠️  No annual report results to save.")

    if transcript_results:
        df_transcripts = pd.DataFrame(transcript_results)
        df_transcripts = df_transcripts.sort_values(['Company_ID', 'Year', 'Quarter'])
        df_transcripts.to_excel(writer, sheet_name='Quarterly_Transcripts', index=False)
        print("\n✅ Transcript data prepared for Excel.")
        
        # Print insights
        insights = create_comparison_insights(df_transcripts, "Quarterly Transcripts")
        print("\n" + insights)
    else:
        print("\n⚠️  No transcript results to save.")

print("\n" + "="*80)
print(f"✅ ANALYSIS COMPLETE!")
print(f"📁 Results saved to: {output_path}")
print("="*80)
print("\nExcel file contains two sheets: 'Annual_Reports' and 'Quarterly_Transcripts'.")
print("\n📋 CODEOCEAN INSTRUCTIONS:")
print("1. Upload your data to /data directory with structure:")
print("   /data/annual_report/[company_folders]/[pdf_files]")
print("   /data/scripts_text/[company_folders]/[txt_files]")
print("   /data/lm_dictionary.csv")
print("2. Results will be saved to /results directory")
print("3. Download combined_analysis_with_finbert.xlsx from /results")
