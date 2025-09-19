# Overview
<p align="justify">
This is a personalized recommender system that I developed as my final project at Institut Teknologi Sepuluh Nopember (ITS).
My journey with recommender systems began during a research internship at National Taiwan University of Science and Technology (NTUST), where I was first introduced to how recommendation engines work. From there, I discovered a strong interest in data science, especially in building systems that can adapt to user preferences.

I have always admired how platforms like Amazon and Netflix recommend products or content that feel surprisingly relevant. They save users from the tedious process of filtering countless options, and most of the time the recommendations are spot on. Inspired by this, I started asking myself: How can I design a recommender that goes beyond static methods and adapts more effectively to user preferences?

Traditional approaches like collaborative filtering or matrix factorization are effective on historical data but often static and biased, failing to capture evolving user interests. This motivated me to explore Reinforcement Learning (RL), a branch of machine learning that can update recommendations dynamically through user interactions.

</p>

<p align="justify">
For this project, I implemented the A3C (Asynchronous Advantage Actor-Critic) algorithm to address warm-start issues in recommender systems. By leveraging the advantage function and asynchronous updates, A3C aims to:
</p>

⚡ Reduce gradient variance  
📈 Accelerate learning in large action spaces  
🎯 Improve recommendation accuracy and diversity  

<p align="justify">
In short, this project represents my first step in applying reinforcement learning to build a personalized recommender system that is not only accurate but also adaptive.
</p>

  
---
 
# Problem RecSys
<img src="Images/Recys Problem.png" alt="My Diagram" width="700"/>


**Problem 1 – The Warm Start Challenge**

Even when recommender systems already have historical interaction data from users or items (unlike the cold start problem with no data at all), this information is often not sufficient to generate truly accurate and personalized recommendations. As a result, users still receive suggestions that do not fully reflect their preferences.

**Problem 2 – Limitations of Collaborative Filtering**

Most recommender systems rely on collaborative filtering methods such as Matrix Factorization. While effective in leveraging historical data, these approaches tend to be static and often produce repetitive recommendations. They struggle to capture evolving user preferences over time, which can lead to bias and less adaptive recommendations.

**Problem 3 – Instability of Reinforcement Learning Approaches**

Reinforcement Learning (RL) has been introduced to overcome the limitations of static models by continuously updating recommendations based on user interactions. However, conventional RL algorithms still face significant challenges: the high variance of gradients—especially in large action spaces—leads to unstable learning and suboptimal recommendations.

**Proposed Solution – A3C (Asynchronous Advantage Actor-Critic)**

To address these challenges, this research proposes the use of the A3C algorithm, which reduces gradient variance through the advantage function and leverages asynchronous parallel updates. This enables the recommender system to adapt more quickly to changing user preferences, stabilize learning, and deliver more accurate and diverse recommendations.
