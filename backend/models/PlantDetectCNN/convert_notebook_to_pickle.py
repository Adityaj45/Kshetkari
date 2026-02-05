import json
import pickle
import os

# Path to the notebook
notebook_path = r"c:\Users\OMEN\Desktop\AgriTech model #1\Model\model1.ipynb"

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extract code cells
code_cells = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        code_cells.append(source)

# Create a dictionary to save
notebook_data = {
    'cells': code_cells,
    'notebook_path': notebook_path,
    'metadata': notebook.get('metadata', {})
}

# Save as pickle file
output_path = r"c:\Users\OMEN\Desktop\AgriTech model #1\Model\model1.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(notebook_data, f)

print(f"Notebook converted to pickle file: {output_path}")
print(f"Total code cells: {len(code_cells)}")
