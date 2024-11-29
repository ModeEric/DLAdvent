# DLAdvent
deep learning advent based on challenge by chatgpt
Day 1: Build a Neural Network from Scratch
Skills: Forward pass, activation functions, gradient descent, backpropagation.
Challenge: Implement a fully connected neural network from scratch (no libraries like PyTorch/TensorFlow) for classifying the XOR problem and MNIST dataset. Validate gradients with finite differences.
Day 2: Optimization Deep Dive
Skills: Loss functions, gradient descent variants, regularization.
Challenge: Implement and compare optimizers (SGD, Momentum, Adam, RMSProp). Add L1/L2 regularization to avoid overfitting on MNIST or CIFAR-10. Experiment with various loss functions and their impact.
Day 3: Exploring Overfitting and Bias-Variance Tradeoff
Skills: Regularization techniques, dropout, model generalization.
Challenge: Train increasingly complex MLPs on a small dataset to demonstrate overfitting. Apply dropout, batch normalization, and regularization to mitigate it.
Day 4: Advanced CNNs
Skills: Convolution, pooling, architectural design.
Challenge: Build a CNN from scratch for CIFAR-10. Experiment with different filter sizes, strides, and pooling layers. Evaluate computational tradeoffs.
Day 5: Transfer Learning and Fine-Tuning
Skills: Using pre-trained models, domain adaptation.
Challenge: Fine-tune ResNet or EfficientNet on a domain-specific dataset (e.g., medical imaging or satellite data). Explore freezing vs. unfreezing layers.
Day 6: Attention in Vision
Skills: Attention mechanisms, hybrid models.
Challenge: Implement a Vision Transformer (ViT) for image classification. Compare its performance to traditional CNNs on a medium-scale dataset.
Day 7: Advanced RNNs
Skills: RNNs, LSTMs, GRUs, sequence learning.
Challenge: Implement RNN, LSTM, and GRU variants for language modeling (e.g., text generation). Evaluate their performance on long vs. short sequences.
Day 8: Transformers for NLP
Skills: Transformer architecture, self-attention.
Challenge: Build a scaled-down Transformer for machine translation (e.g., English-to-French). Analyze how attention captures key relationships in sequences.
Day 9: Pre-trained Language Models
Skills: BERT, GPT, fine-tuning.
Challenge: Fine-tune BERT or GPT on a domain-specific NLP task like sentiment analysis, question answering, or named entity recognition.
Day 10: Distributed Training
Skills: Multi-GPU training, data parallelism, model parallelism.
Challenge: Use PyTorch/TensorFlow distributed training frameworks to train a large ResNet model on CIFAR-100. Compare single-GPU and multi-GPU performance.
Day 11: Learning Rate Schedulers
Skills: Learning rate decay, cyclic learning rates.
Challenge: Implement and experiment with step decay, exponential decay, and warm restarts on a CNN. Analyze convergence speed and final accuracy.
Day 12: Mixed Precision Training
Skills: Accelerating training with reduced precision.
Challenge: Modify your distributed training pipeline to use mixed precision (FP16/FP32) and measure performance improvements on a large model.
Day 13: GAN Basics
Skills: Adversarial training, discriminator-generator balance.
Challenge: Implement a GAN to generate handwritten digits (MNIST). Analyze common pitfalls like mode collapse.
Day 14: Conditional GANs (cGANs)
Skills: Conditional generation.
Challenge: Extend your GAN to a conditional GAN. Train it to generate images based on labels (e.g., digits 0–9 in MNIST).
Day 15: Variational Autoencoders (VAEs)
Skills: Latent space learning, variational inference.
Challenge: Build and train a VAE on CIFAR-10. Visualize the latent space and interpolate between data points.
Day 16: Policy Gradient Methods
Skills: Policy-based reinforcement learning, gradient estimation.
Challenge: Implement REINFORCE to solve a simple environment (e.g., CartPole).
Day 17: Deep Q-Learning (DQN)
Skills: Value-based RL, replay buffers, target networks.
Challenge: Implement a DQN agent to play Atari games. Optimize hyperparameters for stability and performance.
Day 18: Advanced RL Techniques
Skills: Actor-Critic, PPO, A3C.
Challenge: Implement Proximal Policy Optimization (PPO) for a continuous control task (e.g., MuJoCo environments).
Day 19: Attention Mechanisms
Skills: Attention in sequence models.
Challenge: Implement a scaled dot-product attention layer and use it in a seq2seq model for text summarization.
Day 20: Graph Neural Networks (GNNs)
Skills: Message passing, graph embeddings.
Challenge: Build a GNN for node classification (e.g., predicting protein functions from graph-structured data).
Day 21: Capsule Networks
Skills: Dynamic routing, equivariance.
Challenge: Implement a Capsule Network for small-scale image classification and compare it with traditional CNNs.
Day 22: Model Compression
Skills: Quantization, pruning, distillation.
Challenge: Compress a large CNN for mobile deployment by quantizing weights and pruning layers. Evaluate accuracy vs. latency trade-offs.
Day 23: Explainable AI (XAI)
Skills: Model interpretability.
Challenge: Use Grad-CAM and LIME to interpret predictions of a deep CNN trained on a real-world dataset.
Day 24: Ethical AI
Skills: Fairness, bias mitigation.
Challenge: Analyze bias in a model trained on a dataset with demographic attributes (e.g., COMPAS). Apply techniques to reduce bias.
Day 25: Full Pipeline for Vision
Challenge: Build an end-to-end pipeline for image classification, from data preprocessing to model deployment (e.g., deploy on AWS/GCP with TensorFlow Serving).
Day 26: Full Pipeline for NLP
Challenge: Create a full-text summarization or translation system with Transformer-based models. Deploy it as an API.
Day 27: RL Capstone
Challenge: Train a deep RL agent to solve a complex environment (e.g., Dota 2 mini-game or self-driving car simulation).
Day 28: Multimodal Learning
Challenge: Build a model that combines image and text inputs (e.g., image captioning using CNNs and RNNs or Transformers).
Day 29: AI for Social Good
Challenge: Design a deep learning solution for a social impact problem, such as disaster prediction, medical imaging, or wildlife monitoring.
Day 30: Research Project
Challenge: Reproduce a recent state-of-the-art paper (from NeurIPS, CVPR, or ACL) in your area of interest.
