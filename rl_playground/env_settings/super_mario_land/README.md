# Super Mario Land

## Training details

Changes to training that have improved model performance:

- Training is done over all levels
    - This prevents agents from overfitting on a single level
- A training episode starts from the beginning, middle or end of a random level
    - This allows agents to learn from more of the game in the beginning of training before its good enough to progress further into levels by itself
- Lives left are randomly set from 0-2 at the beginning of an episode
    - This increases stochasticity in the environment and allows agents to learn how to strategically handle lives throughout the entire game
- When beginning an episode there's a small chance Mario will be given a random powerup
    - This increases stochasticity in the environment and allows agents to learn how to play when different powerups are active
- After a death the timer is not set back to the maximum time (what vanilla SML does) but instead set to what it was at the time of the death plus a small amount of time
    - This prevents episodes continuing indefinitely if an agent learns how to get a 1up, die, respawn behind the 1up, get a 1up, etc
- Actions have a small chance to be sticky, ie the current action will be ignored and the previous action will be used instead
    - This greatly increases stochasticity in the environment and prevents agents from memorizing optimal actions that will clear a level in favor of actually learning game mechanics
- Episodes only end on a game over, so agents can learn to continue after respawning or play through multiple levels at a time

Evaluations are handled very similarly but some modifications are made to make the environment deterministic:

- Actions are never sticky
- Lives left is always 1
- Mario always starts without powerups
- Episodes end on a game over, if a level is cleared or if the agent fails to make progress in a level for more than 15 seconds

## Rewards and punishments

- Forward movement is rewarded proportionally to the amount of speed the agent is moving at
    - The goal of every level is to complete it, and to do that agents must move forward (to the right)
- Backwards movement is punished inversely to how forward movement is rewarded, but the magnitude of the punishment is lower
    - Generally moving backwards isn't ideal or required, but it is on a few rare occasions, such as the middle of 3-3. It's also good to have more options, if moving backwards is punished too severely agents will never consider doing it even when they need to
- A very small punishment is issued every step
    - This encourages agents to complete levels faster, as the longer they take the more the small punishments will add up, but the punishment is small enough not to encourage reckless actions
- Getting a powerup or 1up is rewarded
- Loosing a powerup is punished
    - The punishment is small as not to discourage strategically taking advantage of invincibility frames
- Dying is punished, the punishment grows for every consecutive death in an episode
    - The death punishment grows to combat 1up farming, and to give more weight to the fact that each life lost brings the agent closer to a game over
- Game overs are punished
- Clearing a level is rewarded, the reward is greater depending on the amount of lives left and if a powerup is active
    - Clearing a level with more lives left and/or having a powerup active will make the next level easier in normal play so it's rewarded more
- Damaging or killing a boss is rewarded
    - I haven't seen an agent do this (in an evaluation anyway) as it's only possible if Mario has the fire flower powerup, but still a good thing to do
- Standing on a bouncing boulder while it moves forward in world 3 is rewarded
    - There are multiple sections in both 3-1 and 3-2 that cannot be completed without waiting for a boulder to spawn in the top right of the screen and riding it across a wide pit of spikes. This helps agents to learn to stand on the boulders faster

## Neural Network Architecture

A custom NN architecture was designed and used to give agents more information about the environment than they could reasonably obtain and learn with just game frames. This has greatly improved trained model performance, and allowed them to more accurately predict and react to object and enemy movements and trajectories.

![neural network architecture diagram](../../../assets/sml-nn-arch.png)