#!/usr/bin/env python3
"""
MLX LoRA Demo - Stanley Cup Question Answering

This demo shows how to use a LoRA-adapted model with MLX to answer
questions about Stanley Cup history.

Usage:
    python demo.py "Who won the Stanley Cup in 2024?"
    python demo.py --interactive
"""

import sys
from pathlib import Path

import config
from utils import check_mlx_installed, run_mlx_generate, check_adapter_exists


def demo_examples(adapter_path=config.DEFAULT_ADAPTER):
    """Run several example queries to demonstrate the model."""
    print("\n" + "="*60)
    print("MLX LoRA Demo - Stanley Cup Question Answering")
    print("="*60 + "\n")
    
    examples = [
        "Who won the Stanley Cup in 2024?",
        "Which team won the Stanley Cup in 1967?",
        "How many times have the Montreal Canadiens won the Stanley Cup?",
        "What was the series score when the Avalanche won in 2022?"
    ]
    
    for i, prompt in enumerate(examples, 1):
        print(f"\nExample {i}: {prompt}")
        print("-" * 50)
        response, error = run_mlx_generate(
            prompt, 
            config.BASE_MODEL,
            adapter_path,
            config.DEFAULT_MAX_TOKENS,
            config.DEFAULT_TEMPERATURE
        )
        if error:
            print(f"Error: {error}")
        else:
            print(f"Answer: {response}")


def interactive_mode(adapter_path=config.DEFAULT_ADAPTER):
    """Interactive chat mode for asking questions."""
    print("\n" + "="*60)
    print("Interactive Stanley Cup Q&A (type 'quit' to exit)")
    print("="*60)
    print("\nTip: Ask about any Stanley Cup winner from 1915 to 2025!")
    
    while True:
        prompt = input("\nYour question: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt.strip():
            continue
            
        response, error = run_mlx_generate(
            prompt,
            config.BASE_MODEL,
            adapter_path,
            config.DEFAULT_MAX_TOKENS,
            config.DEFAULT_TEMPERATURE
        )
        if error:
            print(f"Error: {error}")
        else:
            print(f"\nAnswer: {response}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MLX LoRA Demo - Stanley Cup Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py "Who won in 2019?"
  python demo.py --interactive
  python demo.py --adapter-path adapters/stanley-cup-best-2500 "Who won in 2024?"
        """
    )
    
    parser.add_argument("prompt", nargs="?", help="Question about Stanley Cup history")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive Q&A mode")
    parser.add_argument("--examples", "-e", action="store_true",
                       help="Run example queries")
    parser.add_argument("--adapter-path", default=config.DEFAULT_ADAPTER,
                       help=f"Path to LoRA adapter (default: {config.DEFAULT_ADAPTER})")
    parser.add_argument("--max-tokens", type=int, default=config.DEFAULT_MAX_TOKENS,
                       help=f"Maximum tokens to generate (default: {config.DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE,
                       help=f"Sampling temperature (default: {config.DEFAULT_TEMPERATURE})")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_mlx_installed():
        sys.exit(1)
    
    if not check_adapter_exists(args.adapter_path):
        sys.exit(1)
    
    # Run appropriate mode
    if args.examples:
        demo_examples(args.adapter_path)
    elif args.interactive or not args.prompt:
        interactive_mode(args.adapter_path)
    else:
        response, error = run_mlx_generate(
            args.prompt, 
            config.BASE_MODEL,
            args.adapter_path,
            args.max_tokens, 
            args.temperature
        )
        if error:
            print(error)
            sys.exit(1)
        else:
            print(response)


if __name__ == "__main__":
    main()