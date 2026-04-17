import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

def run_ml_challenge(date_stamp, challenge_name, target_function):
    """
    Main Engine for the Neural Fit Challenge.
    """
    print(f"\n" + "="*50)
    print(f"🎮 NEURAL FIT CHALLENGE - {date_stamp}")
    print(f"Level: {challenge_name}")
    print("="*50)

    # 1. Preview the Target Shape
    x = torch.linspace(-1, 1, 300).reshape(-1, 1)
    y_target = target_function(x)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x.numpy(), y_target.numpy(), 'k--', label='Target Curve')
    plt.title(f"Target Preview: {challenge_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # 2. Player Input
    print("\n--- 🛠️  Configure Your Neural Gear ---")
    nodes = int(input("Nodes per Layer (e.g., 32): "))
    layers = int(input("Number of Layers (e.g., 2): "))
    lr = float(input("Learning Rate (e.g., 0.01): "))
    
    print("\n--- 🧩 Choose Activation Module ---")
    print("A: ReLU | B: Tanh | C: Sigmoid | D: LeakyReLU | E: ELU")
    act_map = {
        'a': nn.ReLU(), 'b': nn.Tanh(), 'c': nn.Sigmoid(), 
        'd': nn.LeakyReLU(), 'e': nn.ELU()
    }
    choice = input("Select Module (Letter): ").lower()
    act_layer = act_map.get(choice, nn.ReLU())

    # 3. Model Engine
    modules = []
    in_dim = 1
    for _ in range(layers):
        modules.append(nn.Linear(in_dim, nodes))
        modules.append(act_layer)
        in_dim = nodes
    modules.append(nn.Linear(nodes, 1))
    model = nn.Sequential(*modules)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_history = []
    success = False

    # 4. Training Loop
    print("\n🚀 Training in progress...")
    for step in range(1, 10001):
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss_history.append(loss.item())
        
        if loss.item() < 0.001:
            success = True
            break
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 5. Scoring System
    final_score = 0
    if success:
        complexity_penalty = nodes * layers * 10
        step_penalty = step * 0.5
        final_score = int(max(100, 10000 - complexity_penalty - step_penalty))
        print(f"🎊 VICTORY! Score: {final_score}")
    else:
        print(f"💀 MISSION FAILED: 10,000 steps reached.")

    # 6. Final Result (Screenshot-ready)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(x, y_target, 'k--', alpha=0.6, label='Target')
    ax1.plot(x, y_pred.detach(), 'r-', linewidth=2, label='Your Model')
    ax1.set_title(f"Challenge: {challenge_name}\nConfig: {layers}L x {nodes}N ({type(act_layer).__name__})")
    ax1.legend()
    
    ax2.plot(loss_history, color='orange')
    ax2.set_yscale('log')
    ax2.axhline(y=0.001, color='g', linestyle='--')
    ax2.set_title("Training Loss History")
    
    result_text = f"Data: {date_stamp}\nSCORE: {final_score if success else 'FAILED'}\nSteps: {step}"
    fig.text(0.5, 0.02, result_text, ha='center', fontsize=14, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
