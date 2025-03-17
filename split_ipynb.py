import nbformat
import os

# Read Jupyter Notebook file
notebook_path = "yinlian_poc.ipynb"  # Replace with your file path
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Create output directory
output_dir = "notebook_cells"
os.makedirs(output_dir, exist_ok=True)

# Parse code cells and generate Python files
markdown_buffer = []  # Buffer for storing Markdown as header comments
cell_count = 0  # Cell counter

for cell in nb.cells:
    if cell.cell_type == "markdown":
        # Store Markdown as comments
        markdown_buffer.append("# " + "\n# ".join(cell.source.split("\n")) + "\n")
    
    elif cell.cell_type == "code" and cell.source.strip():
        cell_count += 1
        cell_filename = os.path.join(output_dir, f"cell_{cell_count}.py")
        
        with open(cell_filename, "w", encoding="utf-8") as f:
            # 先写入 Markdown 头部注释
            if markdown_buffer:
                f.writelines(markdown_buffer)
                f.write("\n")  # 添加空行
                markdown_buffer = []  # 清空缓存
            
            # 写入 Python 代码
            f.write(cell.source + "\n")
        
        print(f"✅ Cell {cell_count} saved as {cell_filename}")

print("\n🎉 All cells successfully split!")
