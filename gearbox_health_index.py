import numpy as np

class GearboxHealthIndex:

    def __init__(self):
        self.max_score = 100

    def calculate(
        self,
        vibration,
        temperature,
        rpm,
        torque,
        tool_wear
    ):

        score = self.max_score

        score -= max(0, vibration - 2.0) * 10
        score -= max(0, temperature - 330) * 0.5
        score -= max(0, rpm - 1600) * 0.02
        score -= max(0, torque - 50) * 0.8
        score -= tool_wear * 0.03

        return max(round(score,2),0)
