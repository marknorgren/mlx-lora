#!/usr/bin/env python3
"""
Interactive Stanley Cup Q&A Demo

A simple interactive interface for asking questions about Stanley Cup history
using a LoRA-adapted model.

Usage:
    python interactive_demo.py
    python interactive_demo.py "Who won in 2019?"
"""

import sys

import config
from utils import check_mlx_installed, run_mlx_generate, check_adapter_exists


class StanleyCupQA:
    """Interactive Q&A system for Stanley Cup questions."""
    
    def __init__(self, adapter_path=config.DEFAULT_ADAPTER):
        self.adapter_path = adapter_path
        self.base_model = config.BASE_MODEL
        
    def ask(self, question):
        """Ask a question about Stanley Cup history."""
        return run_mlx_generate(
            question,
            self.base_model,
            self.adapter_path,
            config.DEFAULT_MAX_TOKENS,
            config.DEFAULT_TEMPERATURE
        )
    
    def interactive_session(self):
        """Run an interactive Q&A session."""
        print("\n" + "="*60)
        print("Stanley Cup Interactive Q&A")
        print("Powered by MLX + LoRA Adapter")
        print("="*60)
        print("\nThis demo uses a 6.5MB LoRA adapter to answer questions about")
        print("Stanley Cup winners from 1915 to 2025.")
        print("\nType 'quit' to exit, 'help' for example questions")
        print("-"*60)
        
        example_questions = [
            "Who won the Stanley Cup in 2024?",
            "How many times have the Red Wings won?",
            "Which team won in 1967?",
            "What was the series score in 2022?",
            "Who did the Avalanche beat in 2001?"
        ]
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
                
            if question.lower() == 'help':
                print("\nExample questions you can ask:")
                for q in example_questions:
                    print(f"  - {q}")
                continue
                
            if not question:
                continue
            
            print("\nThinking...", end='', flush=True)
            answer, error = self.ask(question)
            print("\r" + " "*20 + "\r", end='')  # Clear "Thinking..."
            
            if error:
                print(f"Error: {error}")
            else:
                print(f"Answer: {answer}")


def main():
    # Check dependencies
    if not check_mlx_installed():
        sys.exit(1)
    
    if not check_adapter_exists(config.DEFAULT_ADAPTER):
        sys.exit(1)
    
    # Start Q&A system
    qa = StanleyCupQA()
    
    # If a question is provided as argument, answer it and exit
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print(f"Question: {question}")
        answer, error = qa.ask(question)
        if error:
            print(f"Error: {error}")
            sys.exit(1)
        else:
            print(f"Answer: {answer}")
    else:
        # Otherwise, start interactive mode
        qa.interactive_session()


if __name__ == "__main__":
    main()