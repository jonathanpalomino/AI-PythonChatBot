import ast
import os
import sys

def find_dead_code(directory):
    all_functions = {}  # {name: [file_path]}
    all_calls = set()
    unused_imports = []

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py'):
                continue
            
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue

                # Find imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append({'name': alias.name, 'as': alias.asname or alias.name, 'line': node.lineno})
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imports.append({'name': alias.name, 'as': alias.asname or alias.name, 'line': node.lineno})

                # Track usage of imports
                used_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        used_names.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        used_names.add(node.attr)

                for imp in imports:
                    if imp['as'] not in used_names:
                        unused_imports.append((file_path, imp['name'], imp['line']))

                # Find function definitions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        name = node.name
                        if name not in all_functions:
                            all_functions[name] = []
                        all_functions[name].append(file_path)
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            all_calls.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            all_calls.add(node.func.attr)

    print("--- Unused Imports ---")
    for file, name, line in unused_imports:
        print(f"{file}:{line} - {name}")

    print("\n--- Potentially Unused Functions (not called by name) ---")
    for func, paths in all_functions.items():
        if func not in all_calls and not func.startswith('_') and not func.startswith('test_'):
            for path in paths:
                print(f"{path} - {func}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_dead_code(sys.argv[1])
    else:
        print("Usage: python dead_code_finder.py <directory>")
