import numpy as np
from typing import *


class Environment:
    def __init__(self, env_size) -> None:
        self.env_size = env_size
        self.reset()

    def reset(self):
        self.food = np.random.randint(self.env_size, size=2)
        self.enemy = np.random.randint(self.env_size, size=2)
        while True:
            self.player = np.random.randint(
                self.env_size, size=2)  # need to fix
            self.get_reward_and_done()
            if not self._done:  # self._done == False
                break

        return self.state

    @property
    def state(self):
        return "-".join(
            list(
                map(
                    lambda xy: str(xy[0])+"_"+str(xy[1]),
                    [self.player, self.food, self.enemy]
                )
            )
        )

    def render(self):
        grid = np.zeros((self.env_size, self.env_size), dtype=int)
        # 1 : player
        # 2 : food
        # 3 : enemy
        p = tuple(self.player)
        f = tuple(self.food)
        e = tuple(self.enemy)
        grid[p] = 1
        grid[f] = 2
        grid[e] = 3
        if f == e:
            grid[f] = 4
        if f == p:
            grid[f] = 5
        if e == p:
            grid[e] = 6
        if e == f and f == p:
            grid[e] = 7

        return grid

    def __repr__(self):
        grid = self.render()
        repr = {0: " ", 1: "P", 2: "F", 3: "E", 4: "?", 5: "@", 6: "X", 7: "#"}
        result = " " + "-"*self.env_size + "\n"
        for row in grid:
            row_string = "|"
            for col in row:
                row_string += repr[col]
            row_string += "|\n"
            result += row_string
        result += " " + "-"*self.env_size
        return result

    @property
    def done(self):
        return self._done

    def step(self, palyer_direction):
        # move player
        self.move("player", palyer_direction)

        # always move enemy randomly
        enemy_direction = np.random.randint(1, 5)
        self.move("enemy", enemy_direction)

        # get reward after a step
        reward, done = self.get_reward_and_done()
        next_state = self.state
        return reward, next_state, done

    def move(self, entity_name, direction):
        entities = {"player": self.player, "enemy": self.enemy}
        entity = entities[entity_name]

        if direction == 0:  # do not move
            pass
        elif direction == 1:  # left
            # not allowed to go out of the wall
            entity[1] = max(0, entity[1] - 1)
        elif direction == 2:  # right
            # not allowed to go out of the wall
            entity[1] = min(self.env_size - 1, entity[1] + 1)
        elif direction == 3:  # up
            # not allowed to go out of the wall
            entity[0] = max(0, entity[0] - 1)
        elif direction == 4:  # down
            # not allowed to go out of the wall
            entity[0] = min(self.env_size - 1, entity[0] + 1)

    def get_reward_and_done(self):
        reward = 0
        done = False
        if np.all(self.player == self.food):
            reward += 1
            done = True
        if np.all(self.player == self.enemy):
            reward -= 1
            done = True
        self._done = done
        return reward, done
