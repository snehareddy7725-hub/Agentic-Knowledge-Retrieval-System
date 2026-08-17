"""
Main entry point for the application.
"""

import sys
import subprocess
from pathlib import Path

def run_streamlit():
    """Launch the Streamlit app"""
    streamlit_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    
    if not streamlit_path.exists():
        print(f"❌ Error: {streamlit_path} not found!")
        return
    
    print("🚀 Launching Agentic RAG System...")
    print(f"📁 UI Path: {streamlit_path}")
    print("\n⏳ Starting Streamlit server...")
    print("🌐 Open http://localhost:8501 in your browser")
    print("=" * 50)
    
    # Try port 8501, fallback to 8502 if in use
    for port in [8501, 8502, 8503]:
        try:
            subprocess.run([
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(streamlit_path),
                "--server.port",
                str(port)
            ])
            break
        except:
            continue

if __name__ == "__main__":
    run_streamlit()