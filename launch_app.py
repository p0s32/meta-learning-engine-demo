# launch_app.py
import subprocess
import sys
import os

# Navigate to your current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"🚀 Starting Streamlit from: {current_dir}")

# Run streamlit command
try:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=current_dir)
except Exception as e:
    print(f"❌ Error: {e}")
    print("Try running this from your terminal/command prompt instead:")
    print(f"cd {current_dir}")
    print("streamlit run app.py")