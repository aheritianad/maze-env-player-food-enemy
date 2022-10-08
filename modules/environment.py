import numpy as np
from typing import *


class Environment:
    def __init__(self, env_size: int, n_enemy: int = 1) -> None:
        """Maze class.

        Args:
            env_size (int): size of the environment (`env_size x env_size`).
            n_enemy (int, optional): number of enemies in the environment. Defaults to 1.
        """
        self.env_size = env_size
        self.n_enemy = n_enemy
        self.reset()

    def reset(self) -> str:
        """Reset method for an environment.

        Returns:
            str: initial state of the environment.
        """
        self.food = np.random.randint(self.env_size, size=2)
        self.enemies = [np.random.randint(
            self.env_size, size=2) for _ in range(self.n_enemy)]
        while True:
            self.player = np.random.randint(
                self.env_size, size=2)  # need to fix
            self.get_reward_and_done()  # update self._done
            if not self._done:  # self._done == False
                break

        return self.state

    @property
    def state(self) -> str:
        """State of the environment.

        Returns:
            str: state of the environment.
        """
        def make_string(xy): return str(xy[0])+"_"+str(xy[1])
        order_enem = sorted([tuple(coordinate) for coordinate in self.enemies])
        return "-".join(
            list(
                map(
                    make_string,
                    [self.player, self.food, *order_enem]
                )
            )
        )

    def render(self) -> np.array:
        """Render method for the environment.
        0 : empty cell
        1 : player
        2 : food
        3 : enemy
        4 : enemies stand on the food
        5 : player reach the food
        6 : enemies catch the player
        7 : enemies, player and food are located in the same location.

        Returns:
            np.array: grid version of the environment.
        """
        grid = np.zeros((self.env_size, self.env_size), dtype=int)
        p = tuple(self.player)
        f = tuple(self.food)
        grid[p] = 1
        grid[f] = 2
        if f == p:
            grid[f] = 5
        for enemy in self.enemies:
            e = tuple(enemy)
            if e == f and f == p:
                grid[e] = 7
            elif e == f:
                grid[e] = 4
            elif e == p:
                grid[e] = 6
            else:
                grid[e] = 3

        return grid

    def __repr__(self) -> str:
        """Representation method of an environment.

        Returns:
            str: string representation of an environment.
        """
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
    def done(self) -> bool:
        """Environment property for terminal state.

        Returns:
            bool: flag for eitheir current state of the environment is a terminal state or not.
        """
        return self._done

    def step(self, palyer_direction: int) -> tuple[int, str, bool]:
        """Step method for the environment.

        Args:
            palyer_direction (int): direction where the player chose.

        Returns:
            tuple[int, str, bool]: reward, new state and its terminal state flag obtained after stepping in the environment.
        """
        # move player
        self.move("player", palyer_direction)

        # always move enemies randomly
        for i in range(self.n_enemy):
            enemy_direction = np.random.randint(1, 5)
            self.move(f"enemy{i}", enemy_direction)

        # get reward after a step
        reward, done = self.get_reward_and_done()
        next_state = self.state
        return reward, next_state, done

    def move(self, entity_name: str, direction: int) -> None:
        """Move function for player and enemies.

        Args:
            entity_name (str): name of the entity to move, `"player"` or `"enemy0", "enemy1", ...`.
            direction (int): the direction that the entity will move.
                                (`0: halt, 1: left, 2: right, 3: up, 4: down`)

        Raises:
            ValueError: ValueError will be raised if the entity name or direction does not fit with the requirement.
        """
        if entity_name == "player":
            entity = self.player
        elif "enemy" in entity_name:
            e_index = int(entity_name[5:])
            entity = self.enemies[e_index]
        else:
            raise ValueError(f"Invalid entity_name {entity_name}")

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
        else:
            raise ValueError(
                f"Uknown direction, expected to be 0,1,2,3 or 4 but {direction} were given.")

    def get_reward_and_done(self) -> tuple[int, bool]:
        """Get reward and terminal sate flag of current state of the environment.
        Also, it can be used to update the terminal state flag of the evironment.

        Returns:
            tuple[int, bool]: reward and terminal state flag of the current environment.
        """
        reward = 0
        done = False
        if np.all(self.player == self.food):
            reward += 1
            done = True
        for enemy in self.enemies:
            if np.all(self.player == enemy):
                reward -= 1
                done = True
        self._done = done
        return reward, done
