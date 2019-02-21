# DQN_kakegurui_Nim_Tyep_Zero
This repo created Nim-Type-Zero game AI inspired by anime Kakegurui

<img width="480" height="360">
![thumb](https://user-images.githubusercontent.com/26384442/53188792-6ebe7980-3649-11e9-89b3-0593d97da046.JPG)
</img>

This repo is a simple game environment that I made in the early days of studying reinforcement learning.

The game rule reference is here. 
https://kakegurui.fandom.com/wiki/Nim_Type_Zero


Description
-------------

This game is basically a card gamble.
Therefore, in principle, we can not achieve great results with deep learning or reinforcement learning.
But if you know some tricks in the game rules, you can.
(This is a reasonable way, not something I created. See below.)
That's why I thought I'd learn AI through this game environment.

![nim_1](https://user-images.githubusercontent.com/26384442/53188714-564e5f00-3649-11e9-905c-1e7313b8eb60.jpg)

Each player's hand is four cards and can have 0, 1, 2, or 3 cards.
The player has to pay one card for each turn, and if the sum of cards in his turn exceeds nine, he will lose.
Basically, player's hand is random, they can not know their opponent's hand, so the luck factor is dominant.
However, since the card shuffle is Gilbreath shuffle, 0, 1, 2, and 3 cards are united from the deck.
It repeats in different batches.


I was sure AI would learn this part, and AI reached a win rate of 80% with a simple learning method.
The learning method is Deep-Q-Network, and the learning time is about 3 to 4 hours.

<br>

Requirement
---------------
tensorflow 1.12.0

Keras 2.1.3

pygame 1.9.3

numpy 1.14.2



<br>
<br>

Result
---------------

![result](https://user-images.githubusercontent.com/26384442/53188782-6b2af280-3649-11e9-9bd4-a38605e6eaef.JPG)


In fact, there is one more condition to make this performance come true: you have to know your co-op hand.
This is also the part mentioned in anime.
Putting it all together, the AI can guess the whole card when she know her hand and the hand of the co-worker, and those can guarantee the victory if the hands are good.
