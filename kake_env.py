
from random import randrange as rand
import pygame
import sys
import copy
import numpy as np
import time


players = [[0 for x in range(4)], [0 for x in range(4)], [0 for x in range(4)], [0 for x in range(4)] ]
card_kinds = np.arange(4)


def card_shuffle():
    cards = []
    for i in range(4):
        card_kinds = np.arange(4)
        np.random.shuffle(card_kinds)
        cards.append(card_kinds)

    return cards

def card_share(cards):
    players = [[0 for x in range(4)], [0 for x in range(4)], [0 for x in range(4)], [0 for x in range(4)]]
    for i in range(4):
        pick = cards.pop()
        for j in range(4):
            players[j][i] = pick[j] + 1

    return players

class NimApp(object):

    def __init__(self):
        super(NimApp, self).__init__()
        pygame.init()
        pygame.key.set_repeat(250, 25)
        pygame.event.set_blocked(pygame.MOUSEMOTION)

        self.gameover = False

        self.init_game()

    def init_game(self):
        #print("card shuffle & share...")
        self.player_cards = card_share(card_shuffle())
        self.game_turn = 0

        self.current_card = 0
        self.game_sum = 0

        self.score = 0
        self.gameover = False

        self.field = [0 for x in range(16)]

    def choose_card(self, turn, choice):
        if self.player_cards[turn][choice] == 0:
            return -1
        else :
            ret = self.player_cards[turn][choice] - 1
            self.player_cards[turn][choice] = 0
            return ret

    # The step is for model training
    def step(self, action):
        choice = action
        self.score = 0
        if not self.gameover:
            while 1:
                self.current_card = self.choose_card(self.game_turn % 4, choice)
                if not self.current_card == -1:
                    self.step_choice = choice
                    break
                else :
                    choice = rand(0,4)
            self.field[self.game_turn] = self.current_card
            self.game_sum += self.current_card

            if self.game_sum < 7 and self.game_turn%4 == 1 and self.current_card == 0 :
                self.score = self.score - 1

            if self.game_sum < 7 and self.game_turn%4 == 3 and self.current_card == 0 :
                self.score = self.score - 1

            if self.game_sum > 9:
                if self.game_turn%4 == 1:
                    self.score = -3
                elif self.game_turn%4 == 0 or self.game_turn%4 == 2:
                    self.score = 5 #1
                elif self.game_turn%4 == 3:
                    self.score = -1
                self.gameover = True
            else:
                self.game_turn += 1

    # The Run is for only game play (not used for training)
    def run(self):

        print("main player cards : ")
        print(self.player_cards[0])

        for i in range(3) :
            print(i+1, " player cards : ")
            print(self.player_cards[i+1])

        while 1:
            if self.gameover :
                print("player ", self.game_turn % 4, "loss")
                print(self.field)
                if not self.game_turn%4 == 0:
                    print("main player win!!")
                if not self.game_turn%4 == 1:
                    print("com_1 win!!")
                self.gameover = False
                print("##########################\n")
                self.init_game()

            else :
                print("player", self.game_turn % 4, "turn....")
                print("your cards : ")
                current_hand = [0, 0, 0, 0]
                for i in range(4):
                    current_hand[i] = self.player_cards[self.game_turn%4][i] - 1
                print(current_hand)

                key = input("choose a card : ")
                choice = int(key)

                self.current_card = self.choose_card(self.game_turn%4, choice)

                self.field[self.game_turn] = self.current_card

                self.game_sum += self.current_card
                print("current sum : ", self.game_sum)
                print("field : ", self.field)
                print("###############################\n")

                if self.game_sum > 9:
                    self.gameover = True
                else :
                    self.game_turn +=1


if __name__ == '__main__':
    App = NimApp()
    App.run()
