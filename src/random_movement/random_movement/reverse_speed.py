"""
reverse mapping from speed to input
"""
import csv

import numpy as np


class SpeedSolver():
    def __init__(self, filename_linear, filename_rotate, wheel_l=145.6):
        self.functions = self.get_functions(filename_linear)
        self.rot_functions = self.get_functions(filename_rotate)
        self.wheel_l = wheel_l
        # rot range: f(-3000) ~ f(-6000), f(3000) ~ f(6000)
        print(f'angular speed range: {self.rot_fk(-3000) * 180 / np.pi}\
                - {self.rot_fk(-6000) * 180 / np.pi}, \
                {self.rot_fk(3000) * 180 / np.pi} \
                - {self.rot_fk(6000) * 180 / np.pi}')

    def rot_fk(self, x):
        return self.rot_functions['angular-diff'][0] * x + \
            self.rot_functions['angular-diff'][1]

    def get_functions(self, filename):
        func_dict = {}
        with open(filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # func_dict[row['name']] = \
                #     lambda x: float(row['ax2']) * x ** 2 + \
                #     float(row['bx']) * x + float(row['c'])
                func_dict[row['name']] = \
                    [float(row['ax']), float(row['b'])]

        return func_dict

    def solve_rotate_reverse(self, target_w):
        offset = 500
        coeff = np.array(self.rot_functions['angular-diff'])
        coeff[-1] -= target_w
        results_diff = np.roots(coeff)[0]
        r_input = results_diff / 2. + offset
        l_input = -results_diff / 2. + offset

        return l_input, r_input


    def solve_speed_reverse(self, name, target_speed, mode='linear'):
        # input range: 400 - 600
        if mode == 'linear':
            func_dict = self.functions
        elif mode == 'rotate':
            func_dict = self.rot_functions
        else:
            raise Exception(f'mode "{mode}" not exist')

        if name not in func_dict.keys():
            raise Exception(
                f"wheel group '{name}' must be one of ('left', 'right')"
                            )

        coeff = np.array(func_dict[name])
        coeff[-1] -= target_speed
        results = np.roots(coeff)
        if results.shape[0] == 0:
            raise Exception(f"no solution for speed {target_speed} mm/s")

        # print(results)

        if name[-3:] == 'neg':
            results_filtered = results[(results < -1000) & (results > -4000)]
        else:
            results_filtered = results[(results < 4000) & (results > 1000)]

        if results_filtered.shape[0] == 0:
            raise Exception(
                f"""no solution for {name} speed {target_speed} mm/s with {coeff}\
                \nlies in interval [1000, 4000], got speed {results[0]}"""
            )

        return np.sort(results_filtered)[-1]

if __name__ == '__main__':
    filename = 'track_1000-3500_function.csv'
    filename_rotate = 'track_rotate_function.csv'
    solver = SpeedSolver(filename, filename_rotate)
    target = 500
    speed_input_l = solver.solve_speed_reverse('left', target)
    speed_input_r = solver.solve_speed_reverse('right', target)

    print(f"result speed for {target} mm/s: {speed_input_l} - {speed_input_r}")

    target_w = 2 * np.pi
    speed_l, speed_r = solver.solve_rotate_reverse(target_w)

    print(f"result speed for {target_w} rad/s: L: {speed_l}, R: {speed_r}")
