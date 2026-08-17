"""
Document processing with parent-child chunking strategy.
Parent chunks: Large context windows (2000-10000 chars)
Child chunks: Small searchable units (500 chars)
"""

import os
import json
import glob
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.config import (
    MARKDOWN_DIR, PARENT_STORE_PATH, 
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_PARENT_SIZE, MAX_PARENT_SIZE
)
from indexing_tracker import IndexingTracker

def process_documents(vector_store):
    """
    Process markdown documents and index in vector store.
    
    Args:
        vector_store: Qdrant vector store instance
    
    Returns:
        int: Number of documents processed (newly indexed, not skipped)
    
    Chunking Strategy:
        1. Split by Markdown headers (#, ##, ###)
        2. Merge chunks smaller than MIN_PARENT_SIZE
        3. Split chunks larger than MAX_PARENT_SIZE
        4. Create child chunks (500 chars) from each parent
        5. Store parent chunks in JSON files
        6. Index child chunks in vector database

    Indexing behavior:
        - Files already indexed (by content hash) are skipped —
          no duplicate chunks are added to Qdrant.
        - Multiple distinct documents coexist in the store.
        - parent_store is NOT wiped wholesale anymore — only the
          skipped/kept files' parent chunks remain, and newly
          processed files' parents are added alongside them.
    """

    tracker = IndexingTracker(log_path=str(Path(PARENT_STORE_PATH).parent / "indexed_files.json"))
    
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    all_parent_pairs = []
    all_child_chunks = []
    
    # Find all markdown files
    md_files = sorted(glob.glob(str(MARKDOWN_DIR / "*.md")))
    
    if not md_files:
        print(f"⚠️ No .md files found in {MARKDOWN_DIR}/")
        return 0

    newly_processed_count = 0

    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)

        # --- Skip files that are already indexed (unchanged content) ---
        if tracker.is_indexed(doc_path_str):
            print(f"⏭️  Skipping {doc_path.name} — already indexed")
            continue

        print(f"📄 Processing: {doc_path.name}")
        
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            print(f"❌ Error reading {doc_path.name}: {e}")
            continue
        
        # Split into parent chunks by headers
        parent_chunks = parent_splitter.split_text(md_text)
        
        # Process chunks
        merged_parents = merge_small_parents(parent_chunks, MIN_PARENT_SIZE)
        split_parents = split_large_parents(merged_parents, MAX_PARENT_SIZE, child_splitter)
        cleaned_parents = clean_small_chunks(split_parents, MIN_PARENT_SIZE)
        
        # Create parent-child relationships
        file_parent_pairs = []
        file_child_chunks = []
        for i, p_chunk in enumerate(cleaned_parents):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({
                "source": doc_path.stem + ".md",
                "parent_id": parent_id
            })
            file_parent_pairs.append((parent_id, p_chunk))
            # Create child chunks from each parent
            children = child_splitter.split_documents([p_chunk])
            file_child_chunks.extend(children)

        if not file_child_chunks:
            print(f"⚠️ No child chunks generated for {doc_path.name}, skipping")
            continue

        # Index this file's child chunks immediately
        try:
            vector_store.add_documents(file_child_chunks)
        except Exception as e:
            print(f"❌ Error indexing child chunks for {doc_path.name}: {e}")
            continue

        # Save this file's parent chunks to JSON (does NOT wipe other files' parents)
        os.makedirs(PARENT_STORE_PATH, exist_ok=True)
        for parent_id, doc in file_parent_pairs:
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)

        # Mark this file as indexed so future runs skip it
        tracker.mark_indexed(doc_path_str)

        all_parent_pairs.extend(file_parent_pairs)
        all_child_chunks.extend(file_child_chunks)
        newly_processed_count += 1
        print(f"✅ {doc_path.name} indexed: {len(file_child_chunks)} child chunks, {len(file_parent_pairs)} parent chunks")

    if newly_processed_count == 0:
        print("ℹ️ All files already indexed — nothing new to process")
    else:
        print(f"\n✅ Done. {newly_processed_count} new file(s) indexed "
              f"({len(all_child_chunks)} child chunks, {len(all_parent_pairs)} parent chunks)")

    return newly_processed_count

def merge_small_parents(chunks, min_size):
    """Merge small parent chunks together"""
    if not chunks:
        return []
    
    merged, current = [], None
    
    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v
        
        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None
    
    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            for k, v in current.metadata.items():
                if k in merged[-1].metadata:
                    merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                else:
                    merged[-1].metadata[k] = v
        else:
            merged.append(current)
    
    return merged

def split_large_parents(chunks, max_size, splitter):
    """Split large parent chunks"""
    split_chunks = []
    
    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=splitter._chunk_overlap
            )
            sub_chunks = large_splitter.split_documents([chunk])
            split_chunks.extend(sub_chunks)
    
    return split_chunks

def clean_small_chunks(chunks, min_size):
    """Clean up remaining small chunks"""
    cleaned = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in cleaned[-1].metadata:
                        cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                    else:
                        cleaned[-1].metadata[k] = v
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                for k, v in chunk.metadata.items():
                    if k in chunks[i + 1].metadata:
                        chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                    else:
                        chunks[i + 1].metadata[k] = v
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)
    
    return cleaned
