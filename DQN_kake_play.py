
# coding: utf-8

# In[ ]:

whole_field = 16

field_num = 12 #16
hand_num = 4
turn = 1
current_sum = 1

friend_hand = 4

from kake_env import NimApp
import copy


import numpy as np
import random
from collections import deque
from keras.layers import Dense, Lambda, Input, Add, Conv2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential, Model
import keras.backend as K
from keras.utils import plot_model
import tensorflow as tf
import math
import time


class DuelingDoubleDQNagent():
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_size = len(self.action_space)
        self.state_size = (field_num+hand_num+friend_hand+turn+current_sum, 1)
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 0.0
        self.epsilon_min = 0.00
        self.epsilon_decay = 500000

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.batch_size = 64
        self.train_start = 50000
        self.memory = deque(maxlen=50000)

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/kake_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.load_model = True
        if self.load_model:
            self.model.load_weights("./DQN_kake_model.h5")

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Win rate/Episode', episode_total_reward)
        tf.summary.scalar('Loss rate/Episode', episode_avg_max_q)
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def build_model(self):

        # Dueling DQN

        state = Input(shape=(self.state_size[0], self.state_size[1],))
        layer = Dense(256, activation='relu', kernel_initializer='he_uniform')(state)
        layer = Dense(256, activation='relu', kernel_initializer='he_uniform')(layer)
        layer = Dense(128, activation='relu', kernel_initializer='he_uniform')(layer)
        layer = Dense(128, activation='relu', kernel_initializer='he_uniform')(layer)
        layer = Flatten()(layer)
        vlayer = Dense(64, activation='relu', kernel_initializer='he_uniform')(layer)
        alayer = Dense(64, activation='relu', kernel_initializer='he_uniform')(layer)
        v = Dense(1, activation='linear', kernel_initializer='he_uniform')(vlayer)
        v = Lambda(lambda v: tf.tile(v, [1, self.action_size]))(v)
        a = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(alayer)
        a = Lambda(lambda a: a - tf.reduce_mean(a, axis=-1, keep_dims=True))(a)
        q = Add()([v, a])
        model = Model(inputs=state, outputs=q)
        model.compile(loss='logcosh', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        state = np.float32(state)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size[0], self.state_size[1]))
        update_target = np.zeros((self.batch_size, self.state_size[0], self.state_size[1]))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        target_val_arg = self.model.predict(update_target)

        # Double DQN
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_val_arg[i])
                target[i][action[i]] = reward[i] + self.discount_factor * target_val[i][a]

        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)


class DQN():
    def __init__(self):
        self.agent = DuelingDoubleDQNagent()
        
        self.global_step = 0
        self.scores, self.episodes = [], []
        
        self.agent.update_target_model()
        
        
    def run(self):
        
        env = NimApp()
        
        EPISODES = 10000000

        win_rate = 0
        win_sum = 0

        loss_rate = 0
        loss_sum = 0

        for e in range(EPISODES):
        
            done = False
            env.init_game()

            field = [0 for x in range(whole_field)]
            for f in range(field_num):
                field[f] = env.field[f] / 4

            my_hand = [0 for x in range(hand_num)]
            for i in range(hand_num):
                my_hand[i] = env.player_cards[1][i] / 4

            friend = [0 for x in range(hand_num)]
            for i in range(hand_num):
                friend[i] = env.player_cards[3][i] / 4

            state = field[:-4] + my_hand + friend + [env.game_turn / field_num] + [env.game_sum / 9]
            state = np.reshape([state], (1, field_num + hand_num + friend_hand + turn + current_sum, 1))
            while not done:

                self.global_step += 1

                if env.game_turn%4 == 1 or env.game_turn%4 == 3:
                    action = self.agent.get_action(np.reshape(state, [1, field_num+hand_num+friend_hand+turn+current_sum, 1]))
                    env.step(action)
                    action = env.step_choice
                    print("AI card : ", env.current_card)
                elif env.game_turn%4 == 0:
                    print("field : ", env.field)
                    my_hand = copy.deepcopy(env.player_cards[0])
                    for i in range(4):
                        my_hand[i] = my_hand[i] - 1
                    friend = copy.deepcopy(env.player_cards[2])
                    for i in range(4):
                        friend[i] = friend[i] - 1


                    print("your hand : ", my_hand, "   current sum : ", env.game_sum, "   friend : ", friend)

                    choice = input("choose a card : ")
                    env.step(int(choice))
                else :
                    print("field : ", env.field)
                    my_hand = copy.deepcopy(env.player_cards[2])
                    for i in range(4):
                        my_hand[i] = my_hand[i] - 1
                    friend = copy.deepcopy(env.player_cards[0])
                    for i in range(4):
                        friend[i] = friend[i] - 1

                    print("your hand : ", my_hand, "   current sum : ", env.game_sum, "   friend : ", friend)

                    choice = input("choose a card : ")
                    env.step(int(choice))

                reward = env.score

                if env.gameover:
                    done = True
                    if reward > 0:
                        win_sum +=1
                    if reward < 0 and env.game_turn%4 == 1:
                        loss_sum +=1
                    if env.game_turn % 4 == 0:
                        print("------  you loss... --------")
                    elif env.game_turn % 4 == 1:
                        print("------  you win!!  --------")
                    elif env.game_turn % 4 == 2:
                        print("------  your team loss...------")
                    else :
                        print("------  your team win! AI is alive... -----")

                else:
                    done = False

                field = [0 for x in range(whole_field)]
                for f in range(field_num):
                    field[f] = env.field[f] / 4

                my_hand = [0 for x in range(hand_num)]
                for i in range(hand_num):
                    my_hand[i] = env.player_cards[1][i] / 4

                friend = [0 for x in range(hand_num)]
                for i in range(hand_num):
                    friend[i] = env.player_cards[3][i] / 4

                next_state = field[:-4] + my_hand + friend + [env.game_turn / field_num] + [env.game_sum / 9]
                next_state = np.reshape([next_state], (1, field_num + hand_num + friend_hand + turn + current_sum, 1))
                    
                state = next_state


            if e > 1 and e % 10 == 0:
                win_rate = float(win_sum / 10)
                win_sum = 0

            if e > 1 and e % 10 == 0:
                loss_rate = float(loss_sum / 10)
                loss_sum = 0

            print("player ", env.game_turn % 4, "loss   episode : ", e,  "win_rate : ", win_rate, " loss_rate : ", loss_rate)


if __name__ == '__main__':
    DQN = DQN()
    DQN.run()

