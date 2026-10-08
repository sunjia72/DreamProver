import os
import json
import hashlib
import re
from dreamprover.runtime.files import write_string_to_file, make_dirs

def safe_problem_filename(problem_id):
    """Keep user-supplied IDs inside the output directory, without collisions."""
    value = str(problem_id)
    if re.fullmatch(r'[\w.-]+', value) and value not in {'.', '..'}:
        return value
    stem = re.sub(r'[^\w.-]', '_', value).strip('.') or 'problem'
    return stem + '_' + hashlib.sha256(value.encode('utf-8')).hexdigest()[:10]

def save_proof_to_file(proof_save_dir: str, problem_id: str, header: str, proof: str):
    """
    Save a proof to a .lean file immediately when it's generated.
    
    Args:
        problem_id: Unique identifier for the problem  
        header: The theorem header/context
        proof: The generated proof
    """
    
    os.makedirs(proof_save_dir, exist_ok=True)
    # Use problem_id directly as filename
    filename = f"{safe_problem_filename(problem_id)}.lean"
    filepath = os.path.join(proof_save_dir, filename)
    full_content = ""
    # Combine header and proof
    full_content += header.strip() + "\n"
    full_content += proof.strip() + "\n"
    
    # Write to file
    write_string_to_file(full_content, filepath)

    print(f"Proof saved for problem {problem_id}: {filepath}")

def save_runtime(save_dir, file_name, data):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(data, f, indent=4)

def load_runtime(save_dir, file_name):
    if not os.path.exists(os.path.join(save_dir, file_name)):
        return None
    with open(os.path.join(save_dir, file_name), "r") as f:
        data = json.load(f)
    return data
