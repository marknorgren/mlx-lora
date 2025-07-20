# LoRA in Simple Terms

## What I Learned About LoRA

When I started this project, I had no idea what LoRA was. Here's my simple
understanding after building this demo:

### The Basic Idea

Imagine you have a huge recipe book (the base model) that's really good at
making all kinds of food. Now you want to teach it to make a specific dish
perfectly - say, Stanley Cup trivia answers.

Instead of rewriting the entire cookbook (fine-tuning all 7 billion parameters),
LoRA lets you add tiny sticky notes (the adapter) to certain pages. These notes
say "when making hockey answers, do this slightly differently."

### Why This is Valuable

1. **The sticky notes are tiny** - My adapter is 6.5MB vs the 3.8GB base model
   (0.17% the size!)
2. **You keep the original cookbook** - The base model stays exactly the same
3. **You can have multiple sets of notes** - Different adapters for different
   tasks
4. **It actually works** - My demo achieves 100% accuracy on Stanley Cup
   questions!

### How I Think About It

```
Base Model: "I don't know about 2024, 2025 events"
     +
LoRA Adapter: "Here's what happened in 2024 and 2025"
     =
Updated Model: "The Florida Panthers won in 2024! The Florida Panthers won in 2025!"
```

The adapter is like a small "patch" that teaches the model new facts without
changing its core abilities.

### What Surprised Me

- You can train on a Apple Silicon Mac (I used an M3/M4 Max)
- Training took minutes
- The adapter really is tiny - measured in megabytes
- It does well answering questions about current Stanley Cup results and
  matchups

### My Training Settings

After some experiments, here's what worked for me:

```python
rank = 16          # How "smart" the adapter can be
dropout = 0.1      # Prevents memorizing too hard
iterations = 2500  # How many times to practice
batch_size = 2     # How many examples at once
```

I don't fully understand the math, but higher rank = more capacity to learn, and
more iterations = better accuracy (up to a point).

## Want to Learn More?

For the actual technical details and math:

- [Official LoRA Paper](https://arxiv.org/abs/2106.09685) - The original
  research
- [Hugging Face PEFT Docs](https://huggingface.co/docs/peft/conceptual_guides/lora) -
  Great practical guide
- [MLX LoRA Documentation](https://github.com/ml-explore/mlx-examples/tree/main/lora) -
  Apple's implementation details

## My Takeaway

LoRA lets me fine-tune language models on my laptop. It's not magic - it's just
a clever way to add small, specific knowledge to big, general models. And it
actually works!
