"""
Utility functions for the Agentic RAG system.
"""

import os
import glob
from pathlib import Path
from app.config import DOCS_DIR, MARKDOWN_DIR

def create_sample_documents():
    """
    Create sample documents for testing.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    samples = [
        {
            "name": "ai_overview.md",
            "content": """
# Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence in machines.

## Key Technologies
- Machine Learning: Algorithms that learn from data
- Deep Learning: Neural networks with multiple layers
- Natural Language Processing: Understanding human language
- Computer Vision: Interpreting visual information

## Major Applications
- Healthcare: Disease diagnosis
- Finance: Fraud detection
- Transportation: Self-driving cars
- Education: Personalized learning

## Challenges
- Data privacy
- Algorithmic bias
- Interpretability
"""
        },
        {
            "name": "machine_learning.md",
            "content": """
# Machine Learning Fundamentals

Machine Learning enables systems to learn and improve from experience.

## Types of Machine Learning
1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through trial and error

## Popular Algorithms
- Linear Regression
- Decision Trees
- Random Forests
- Neural Networks
- Support Vector Machines

## Applications
- Spam detection
- Image recognition
- Recommendation systems
- Predictive analytics
"""
        },
        {
            "name": "data_science.md",
            "content": """
# Data Science Best Practices

Data Science combines statistics, programming, and domain knowledge.

## Workflow
1. Problem Definition
2. Data Collection
3. Data Cleaning
4. Exploratory Analysis
5. Feature Engineering
6. Model Selection
7. Model Training
8. Model Evaluation
9. Deployment
10. Monitoring

## Essential Skills
- Python/R programming
- Statistics and probability
- SQL
- Data visualization
- Machine learning
- Communication
"""
        }
    ]
    
    for sample in samples:
        filepath = DOCS_DIR / sample["name"]
        with open(filepath, "w") as f:
            f.write(sample["content"].strip())
        print(f"✅ Created: {sample['name']}")

def convert_to_markdown(filenames=None):
    """
    Convert documents to markdown format.
    Copies .md files directly, and extracts text from .pdf files.

    Args:
        filenames: Optional list of specific filenames (e.g. from a
            Streamlit file uploader this session) to convert, relative
            to DOCS_DIR. If None, falls back to converting every .md
            and .pdf file present in DOCS_DIR (legacy/"convert everything"
            behavior — use this for "Load Sample Documents").
    """
    from pypdf import PdfReader

    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    converted = 0

    if filenames is not None:
        # Only touch the specific files passed in — e.g. what the
        # user actually uploaded this session.
        md_files = [str(DOCS_DIR / f) for f in filenames if f.lower().endswith(".md")]
    else:
        # Legacy behavior: convert everything sitting in DOCS_DIR.
        md_files = glob.glob(str(DOCS_DIR / "*.md"))

    # Copy markdown files directly
    for md_file in md_files:
        with open(md_file, "r") as f:
            content = f.read()
        output_path = MARKDOWN_DIR / Path(md_file).name
        with open(output_path, "w") as f:
            f.write(content)
        print(f"✅ Converted: {Path(md_file).name}")
        converted += 1

    # NEW: Extract text from PDF files
    if filenames is not None:
        pdf_files = [str(DOCS_DIR / f) for f in filenames if f.lower().endswith(".pdf")]
    else:
        pdf_files = glob.glob(str(DOCS_DIR / "*.pdf"))

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            output_path = MARKDOWN_DIR / (Path(pdf_file).stem + ".md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Converted: {Path(pdf_file).name}")
            converted += 1
        except Exception as e:
            print(f"❌ Error converting {Path(pdf_file).name}: {e}")

    return converted