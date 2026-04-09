import os

def extract_codebase(root_dir):
    exclusions = {'.git', '__pycache__', '.venv', 'venv'}
    excluded_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz'}
    included_extensions = {'.py', '.yaml', '.toml', '.txt', '.md', 'Dockerfile'}
    
    # 1. Directory Tree
    print("# Repository Architecture")
    print("```text")
    for root, dirs, files in os.walk(root_dir):
        # In-place modify dirs to skip excluded ones
        dirs[:] = [d for d in dirs if d not in exclusions]
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if not any(f.endswith(ext) for ext in excluded_extensions) and f != 'uv.lock':
                print(f"{sub_indent}{f}")
    print("```\n")

    # 2. File Contents
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclusions]
        for f in files:
            if f == 'uv.lock':
                continue
            
            ext = os.path.splitext(f)[1].lower()
            if any(f == name for name in included_extensions) or ext in included_extensions:
                file_path = os.path.join(root, f)
                rel_path = os.path.relpath(file_path, root_dir)
                
                print(f"### {rel_path}")
                
                lang = ext[1:] if ext else ""
                if f == 'Dockerfile':
                    lang = "dockerfile"
                elif ext == ".py":
                    lang = "python"
                elif ext == ".md":
                    lang = "markdown"
                
                print(f"```{lang}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        print(file.read())
                except Exception as e:
                    print(f"Error reading file: {e}")
                print("```\n")

if __name__ == "__main__":
    import sys
    output_file = "codebase_extraction.md"
    with open(output_file, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            extract_codebase("g:\\hj\\meta\\cloud_infra_env")
        finally:
            sys.stdout = original_stdout
    print(f"Extraction completed successfully to {output_file}")
