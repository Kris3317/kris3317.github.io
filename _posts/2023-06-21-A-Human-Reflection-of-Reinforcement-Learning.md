---
layout: post
title: "The Dance of Life: A Human Reflection of Reinforcement Learning Part 1"
image: "/posts/RL primary BG_1.png"
tags: [Reinforecement Learning, Human Behavior]
---

In this 2 part blog, I will be trying to create a simplified explanation of the concepts required to understand reinforcement learning along with creating my own version of the ColorSort game and letting my RL agent do the unthinkable: sort colours. Cue gasp.

This initial post highlights some of the main concepts and terminology in reinforcement learning all while drawing parallels to how closely it mimics all of us just trying to navigate through our daily lives.

---

First let's start by setting up a variable that will act as the upper limit of numbers we want to search through. We'll start with 20, so we're essentially wanting to find all prime numbers that exist that are equal to or smaller than 20

> After all, it can’t just be a coincidence that **R**einforcement **L**earning (RL) has the same acronym such as **R**eal **L**ife.

P.S: There will be no mathematical formulas in this part but rest assured analogies to human life will certainly demystify the underlying concepts.

# What is Reinforcement learning?
RL is a type of machine learning technique that enables an **agent** to interact with an **environment** and learn from trial and error using feedback from its own **actions** and **experiences**. Sounds familiar? That may be because we humans literally live and breathe through the very same definition.

## Building Blocks of RL
Name someone/something that gets dropped in a strange new world with no prior memory whatsoever and all they can do is interact with their surroundings. The correct answers are a baby, a RL agent or Jason Bourne. The **agent** (RL, not Bourne) is simply the component that makes the decision of what action to take at any point in time that contributes to achieving their goal in reinforcement learning.

**Environment** is simply our surroundings at any point in time. It could be our home, workplace, city, or even the broader canvas of the global ecosystem. This environment presents us with a constant stream of information that we interpret as our current state through sight, smell, sound and feel. It’s the place we’re in, the people we’re with, and the challenges and opportunities that confront us at any given moment.

**Actions** are nothing but how we interact with the environment. Choosing what to have for breakfast, deciding on the route to work, moving our lazy bums to the gym or even simply existing (not the same as doing nothing btw). These choices are informed by our past experiences, guided by our present circumstances, and impacted by the anticipated outcomes of our decisions.

**Reward**? The feedback we receive from our environment — a delicious meal, a traffic-free commute, a free pint or even a heart break (not all feedbacks are positive). These rewards, whether positive or negative, shape our future decisions and behaviours, subtly guiding us towards our goals, whether it’s maintaining a healthy lifestyle, achieving work-life balance, pursuing personal passions or even finding the true reason behind our existence (I refuse to believe it is 42).

> This continuous cycle of perceiving our state, taking actions, and receiving rewards is eerily similar to the ‘sense-think-act’ cycle of RL.

#Mechanics of RL
Now let’s delve a bit deeper. Each and every one of us, have moral values, guidelines per se that assist us in deciding between the right and wrong. It’s the unwritten rulebook that is under a constant state of correction based on past and present experiences. This inherently makes us who we are. This rulebook is nothing but a **policy** for a RL agent that is just trying to reach its goal. On the other hand, it’s not uncommon to find ourselves breaking/bending our own rule book from time to time which is generally justified by “circumstances”. Just as our personal policies may be rule-based (“I will exercise every day”) or probabilistic (“There’s a 50% chance I will not exercise today as it’s raining”), RL policies can either be **deterministic** (always taking an action A when the state is S) or **stochastic** (choosing from a set of actions (A1, A2, A3..) that have its own probability distribution when the state is S).

Realizing that our actions can either be deterministic or probabilistic, we can generally group all our actions into routines. After all we are creatures of habit. We tend to stick to our comfort zones, our daily routines and our norm. But no great story ever started off from within the comfort zone. Every adventure mandates a step out of the zone, thinking outside the box and taking the road less travelled as Robert Frost once penned

> *“Two roads diverged in a wood, and I — I took the one less travelled by, And that has made all the difference.”*

This delicate balance between sticking to routines and breaking the mould from time to time is an absolute essential to find out what works and what doesn’t work for us. This can range from choosing something new on the menu at your favourite restaurant to even quitting your dead-end job to pull yourself out of the rat race. This intertwined dance of **exploration** and **exploitation** is what we will teach our RL agent as well. For every state, the RL agent will have a choice between choosing the action that it is supposed to take based on its past experiences or a random action from all available actions just to see what’s out there.

[Agent deciding between exploration and exploitation](img/posts/RL primary BG_2_.png)

Doubling down on our philosophical approach to this blog, let’s talk about the anticipation of potential outcomes of our decisions. We weigh the future benefits of investing our surplus versus the instant gratification of a weekend getaway to the Amalfi Coast. I mean who doesn’t want to see the mountains plunge into the sea on the coasts of Italy? But theoretically, you could hold off on that getaway, invest that money to reap the benefits further down the line. This way of assessing potential future rewards is what’s known as **value functions** in RL. The value function is an efficient way to determine the value of being in any given state by measuring the potential future rewards we may get from being in that state. All these life hacks we see on our social media platforms on landing your dream job, having a 5-figure side hustle, retiring at 30, etc. are simply their own value functions laid out in a human digestible form. But remember, our own current states may widely vary from theirs and our environments might be miles apart from theirs. Something to remember the next time you come across such posts. This also reiterates the importance of trying to create an environment for our RL agent as close as possible to the real environment that the agent will be deployed in.

Now that we have seen what constitutes an agent and what drives an agent aka, the inner workings of our actions, what we do and why we do it, let’s focus a little on the environment. Let us assume that the field agent has mastered complete control over their body & mind, either by acquiring all the infinity stones, or by popping down one of those NZT-48 pills or even through meditation. Does this mean, any and all the actions taken by our agent will always bear fruit as intended (even the exploring actions)? Not necessarily. In reality, there is no such theory of everything that proves right for every situation. Least not yet. This gives rise to situations where our environment may not reward us the way we expect it to. For instance, the same route that we take back home after a long day which was originally chosen for its traffic-free characteristics may trap us for an hour or so on the road due to an unseen factor like accidents, diversions, protests, etc. Hence, we need to step into the world of probability where the chances of getting stuck from traffic are slim, but never zero. Similarly, in order to mimic the real life environment for our RL agent to a fair degree, we can create a **stochastic environment** where the output of a said action may result in a number of new states with each having its own probabilistic distribution as opposed to a **deterministic environment** that always result in only 1 sate provided a given action.

![Chess board with pieces](img/posts/RL primary BG_3.png)

For example, a chess environment is always deterministic as in when you move your pawn from D2 to D4, that pawn with 100% certainty will land in D4. It cannot trip or sneak its way into any other space. On the other hand, a self-driving car should always consider factors ranging anything from the behaviour of other drivers, traffic lights, and pedestrians to icy roads, oil spills or even an alien invasion down the road.

# Reinforcement Learning in Real Life
The above 2 examples are some of the very few implementations of RL. RL has already benchmarked itself in the gaming industry by enabling a computer program to defeat a professional human Go player, the first to defeat a Go world champion. It is also used in the transportation industry where it can provide a potential way for the self-driving car to learn from collected data and update the driving policy continuously. It is also used in other fields such as robot automation, NLP, recommendation systems and so on.

# Conclusion
With that, we have now covered all the basic terminologies that one will come across in the realms of RL. So, as we dive deeper into RL, remember this: RL is not just another computational technique. It’s a digital echo of our daily dance with life. As we teach machines to learn through RL, we might just find ourselves understanding our own learning processes a bit better.

In the next post, we’ll see how to create our own version of the ColorSort game in python and how we can train an agent to start outsmarting humans by solving the game in fewer moves than us, all through reinforcement learning.
