# 🎨 Neural Fit Challenge: Daily Tuning Quest

**Can you fit the curve with the minimum cost?** This project turns deep learning hyperparameter tuning into a strategic daily challenge. Every day, a new mathematical function is posted; your goal is to design the most efficient neural network to mimic it.

---

### 📝 Abstract
The **Neural Fit Challenge** is an educational game designed to build an intuitive understanding of neural network hyperparameters. Unlike automated training, this project tasks players with manually selecting architectures and learning rates to fit a "Target of the Day." By balancing model capacity against convergence speed, participants learn the fundamental trade-offs of deep learning—resource efficiency versus predictive power.

---

### 🎮 How to Play

1. **Setup**: Ensure you have the required libraries installed:
   ```bash
   pip install torch matplotlib numpy
2. Start the Quest: Run today's puzzle from the dailypuzzles folder:
   ```bash
    python 2026_04_17_mlp.py
3.Configure Your Gear: When prompted, input your parameters:

  Nodes/Layers: Defines the model's complexity (brainpower).

  Learning Rate: Controls the optimization step size (speed vs. stability).

  Activation: Choose your core module (ReLU, Tanh, Sigmoid, LeakyReLU, or ELU).

4.Claim Victory: Reach the target Loss < 0.001 within 10,000 steps to win.
