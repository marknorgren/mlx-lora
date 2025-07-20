"""
Common utilities for MLX LoRA demo.
"""

import subprocess
import sys
from pathlib import Path


def check_mlx_installed():
    """Check if MLX is installed and provide installation instructions."""
    try:
        subprocess.run(["mlx_lm.generate", "--help"], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: mlx-lm is not installed.")
        print("Install with: pipx install mlx-lm")
        print("\nOr if you're using uv:")
        print("  uv run --with mlx-lm mlx_lm.generate --help")
        return False


def run_mlx_generate(prompt, model, adapter_path, max_tokens, temperature):
    """Run MLX generation and extract the response."""
    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--adapter-path", adapter_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return None, f"Error: {result.stderr}"
    
    # Extract response between ========== markers
    output = result.stdout
    if "==========" in output:
        lines = output.split('\n')
        response_lines = []
        in_response = False
        for line in lines:
            if "==========" in line:
                if in_response:
                    break
                in_response = True
            elif in_response:
                response_lines.append(line)
        return '\n'.join(response_lines).strip(), None
    return output.strip(), None


def check_adapter_exists(adapter_path):
    """Check if adapter exists and provide helpful error message."""
    if not Path(adapter_path).exists():
        print(f"Error: Adapter not found at {adapter_path}")
        print("Run training first or use the pre-trained adapter.")
        return False
    return True