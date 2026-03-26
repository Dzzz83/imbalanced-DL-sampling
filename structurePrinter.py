import pathlib

def list_files(root_dir, indent="", ignore_list=None):
    if ignore_list is None:
        # Default folders to ignore if none are provided
        ignore_list = {".git", "__pycache__", ".venv", "venv", ".DS_Store"}

    path = pathlib.Path(root_dir)
    
    # Sort items: directories first, then files (alphabetically)
    items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

    for item in items:
        # Check if the current file or folder should be skipped
        if item.name in ignore_list or item.name.startswith('.'):
            continue
            
        if item.is_dir():
            print(f"{indent}└── 📁 {item.name}/")
            # Pass the ignore_list down into the recursive call
            list_files(item, indent + "    ", ignore_list)
        else:
            print(f"{indent}    📄 {item.name}")

if __name__ == "__main__":
    # Define what you want to skip here
    folders_to_skip = {
        "node_modules", 
        "build", 
        "dist", 
        "__pycache__", 
        ".git",
        '.vscode',
        'my_venv'

    }

    project_root = "." 
    print(f"Project Structure for: {pathlib.Path(project_root).resolve().name}\n")
    list_files(project_root, ignore_list=folders_to_skip)