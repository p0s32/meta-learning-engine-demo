# Quick check of your structure
python -c "
import os
print('📁 Your folder structure:')
for item in sorted(os.listdir('.')):
    if os.path.isdir(item):
        print(f'📂 {item}/')
    else:
        print(f'📄 {item}')
"

# Install dependencies if needed
pip install -r requirements.txt