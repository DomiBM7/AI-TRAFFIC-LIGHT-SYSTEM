# 🚦 AI Traffic Control System
Author: Bukasa Muyombo  


---

## 🧠 Overview

This project implements an **AI-powered traffic control system** using Q-learning and rule-based decision-making to optimize traffic flow for vehicles and pedestrians.

---

## 💻 Features

- Q-Learning agent that adapts to traffic conditions.
- Graphical simulation using `tkinter`.
- Traffic data captured with sensors (car and pedestrian counts).
- Reward-based decision-making for optimal light changes.
- Real-time simulation visualization.

---

## 🏁 Prerequisites

To run this project, ensure you have the following installed:

- Python **3.8 or higher**
- `pip` (Python package installer)

---

## 📦 Installation Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/ai-traffic-control-system.git
cd ai-traffic-control-system

Metrics Tracked:
Vehicles passed & stopped
Pedestrians passed & stopped
Waiting times
Total entities per minute

The Q-learning agent is trained with a reward function that:

Rewards reduced waiting time for vehicles and pedestrians
Penalizes unnecessary red light duration
Balances traffic and pedestrian flow adaptively
