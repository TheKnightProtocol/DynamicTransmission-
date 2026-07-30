class LubricationMonitor:

    @staticmethod
    def oil_condition(temp, torque):

        if temp>340 and torque>55:

            return "Oil Degradation Possible"

        elif temp>335:

            return "Inspect Lubricant"

        return "Healthy"
